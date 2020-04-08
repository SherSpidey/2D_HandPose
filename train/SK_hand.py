import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from SetValues import SV
import model_units_funs


class SK_Model(object):
    def __init__(self, input_size, heatmap_size, batch_size,sk_index, stages=3, joints=21):
        self.joints = joints
        self.SK = np.array(sk_index)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.stages = stages
        self.batch_size=batch_size
        self.stage_heatmap = []
        self.stage_loss = [0] * stages


        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(None, self.input_size, self.input_size, 3),
                                                name='input_placeholder')

        self.heatmap_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None, self.heatmap_size, self.heatmap_size, self.joints),
                                                  name='heatmap_placeholder')

    def build_model(self):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('sub_stages'):
                net = slim.conv2d(self.input_placeholder, 64, [3, 3], scope='sub_conv1')
                net = slim.conv2d(net, 64, [3, 3], scope='sub_conv2')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool1')
                net = slim.conv2d(net, 128, [3, 3], scope='sub_conv3')
                net = slim.conv2d(net, 128, [3, 3], scope='sub_conv4')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool2')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv5')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv6')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv7')
                net = slim.conv2d(net, 256, [3, 3], scope='sub_conv8')
                net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='sub_pool3')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv9')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv10')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv11')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv12')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv13')
                net = slim.conv2d(net, 512, [3, 3], scope='sub_conv14')

                self.sub_stage_img_feature = slim.conv2d(net, 128, [3, 3], scope='sub_stage_img_feature')

            for stage in range(1,self.stages+1):
                center_map=self._get_center(stage)
                self._sk_layers(stage,center_map)

    def _get_center(self, stage):
        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('stage_' + str(stage)):
                if stage != 1:
                    self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
                                                         self.sub_stage_img_feature],
                                                        axis=3)
                else:
                    self.current_featuremap = self.sub_stage_img_feature
                conv1 = slim.conv2d(self.current_featuremap, 512, [1, 1], scope='conv1')
                conv2 = slim.conv2d(conv1, self.joints, [1, 1], scope='conv2')
                kps_0 = slim.conv2d(conv2, 1, [1, 1], scope='key_points_0')
        return kps_0

    def _sk_layers(self, stage, center_map):

        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('stage_' + str(stage),reuse=tf.AUTO_REUSE):
                reson_map = tf.concat([center_map, self.current_featuremap], axis=3)
                mid_net = slim.conv2d(reson_map, 128, [7, 7], scope='midnet1_1')
                mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet1_2')
                mid_net = slim.conv2d(mid_net, 128, [1, 1], scope='midnet1_3')
                key_points = slim.conv2d(mid_net, 5, [1, 1], scope='key_points_1')
                heatmap=tf.concat([center_map, key_points], axis=3)
                for i in range(1, 4):
                    reson_featuremap = tf.concat([key_points,
                                                       self.current_featuremap],
                                                      axis=3)
                    mid_net = slim.conv2d(reson_featuremap, 128, [7, 7], scope='midnet_'+str(i+1)+'1')
                    mid_net = slim.conv2d(mid_net, 128, [7, 7], scope='midnet_'+str(i+1)+'2')
                    mid_net = slim.conv2d(mid_net, 128, [1, 1], scope='midnet_'+str(i+1)+'3')
                    key_points = slim.conv2d(mid_net, 5, [1, 1], scope='key_points_'+str(i+1))
                    heatmap=tf.concat([heatmap, key_points], axis=3)

            mid_map=heatmap[:,:,:,self.SK[0]][:,:,:,np.newaxis]
            for i in range(1,21):
                mid_map=tf.concat([mid_map,heatmap[:,:,:,self.SK[i]][:,:,:,np.newaxis]],axis=3)
            self.stage_heatmap.append(mid_map)

    def build_loss(self, lr, lr_decay_rate, lr_decay_step):
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.heatmap_placeholder,
                                                       name='l2_loss') / self.batch_size
            #tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            #tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            #tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer='Adam')
        #self.merged_summary = tf.summary.merge_all()

    def load_weights_from_file(self,weight_file_path, sess, finetune=True):
        pass



