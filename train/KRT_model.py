import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class KRT(object):
    def __init__(self,input_size,heatmap_size,batch_size,sk_index,joints=21):
        self.input_size=input_size
        self.heatmap_size=heatmap_size
        self.batch_size=batch_size
        self.sk=sk_index
        self.joints=joints
        self.heatmap=[]

        self.input_placeholder=tf.placeholder(dtype=tf.float32,
                                                shape=(None, self.input_size, self.input_size, 3),
                                                name='input_placeholder')
        self.heatmap_placeholder=tf.placeholder(dtype=tf.float32,
                                                  shape=(None, self.heatmap_size, self.heatmap_size, self.joints),
                                                  name='heatmap_placeholder')
    def build_model(self):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('pre_work'):
                net = slim.conv2d(self.input_placeholder, 64, [3, 3], scope='pre_conv1')
                net = slim.conv2d(net, 64, [3, 3], scope='pre_conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pre_pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='pre_conv3')
                net = slim.conv2d(net, 128, [3, 3], scope='pre_conv4')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pre_pool2')
                net = slim.conv2d(net, 256, [3, 3], scope='pre_conv5')
                net = slim.conv2d(net, 256, [3, 3], scope='pre_conv6')
                net = slim.conv2d(net, 256, [3, 3], scope='pre_conv7')
                net = slim.conv2d(net, 256, [3, 3], scope='pre_conv8')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pre_pool3')
                net = slim.conv2d(net, 512, [3, 3], scope='pre_conv9')
                net = slim.conv2d(net, 512, [3, 3], scope='pre_conv10')
                net = slim.conv2d(net, 512, [3, 3], scope='pre_conv11')
                net = slim.conv2d(net, 512, [3, 3], scope='pre_conv12')
                net = slim.conv2d(net, 512, [3, 3], scope='pre_conv13')
                net = slim.conv2d(net, 512, [3, 3], scope='pre_conv14')

                self.raw_img_feature = slim.conv2d(net, 128, [3, 3], scope='raw_img_feature')

            with tf.variable_scope('root'):
                self.current_featuremap=self.raw_img_feature
                mid_net = slim.conv2d(self.current_featuremap, 128, [7, 7], scope='mid1')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='mid2')
                mid_net = slim.conv2d(mid_net, 256, [3, 3], scope='mid3')
                mid_net = slim.conv2d(mid_net, self.joints, [1, 1], scope='mid4')
                root_kp = slim.conv2d(mid_net, 1, [1, 1], scope='key_points_0')

            self._sk_layers(root_kp)

    def _sk_layers(self, root_kp):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('tree'):
                reson_map = tf.concat([root_kp, self.current_featuremap], axis=3)
                mid_net = slim.conv2d(reson_map, 128, [7, 7], scope='midnet1_1')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet1_2')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet1_3')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet1_4')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet1_5')
                mid_net = slim.conv2d(mid_net, self.joints, [1, 1], scope='midnet1_6')
                key_points = slim.conv2d(mid_net, 5, [1, 1], scope='key_points_1')
                heatmap = tf.concat([root_kp, key_points], axis=3)
                for i in range(1, 4):
                    reson_featuremap = tf.concat([key_points,
                                                  self.current_featuremap],
                                                 axis=3)
                    mid_net = slim.conv2d(reson_featuremap, 128, [7, 7], scope='midnet_' + str(i + 1) + '1')
                    mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet_' + str(i + 1) + '2')
                    mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet_' + str(i + 1) + '3')
                    mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet_' + str(i + 1) + '4')
                    mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet_' + str(i + 1) + '5')
                    mid_net = slim.conv2d(mid_net, self.joints, [1, 1], scope='midnet_' + str(i + 1) + '6')
                    key_points = slim.conv2d(mid_net, 5, [1, 1], scope='key_points_' + str(i + 1))
                    heatmap = tf.concat([heatmap, key_points], axis=3)

            mid_map = heatmap[:, :, :, self.sk[0]][:, :, :, np.newaxis]
            for i in range(1, 21):
                mid_map = tf.concat([mid_map, heatmap[:, :, :, self.sk[i]][:, :, :, np.newaxis]], axis=3)
            self.heatmap.append(mid_map)

    def build_loss(self, lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):
        self.loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        with tf.variable_scope('loss'):
            self.loss = tf.nn.l2_loss(self.heatmap - self.heatmap_placeholder,name='l2_loss') / self.batch_size

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=optimizer)

