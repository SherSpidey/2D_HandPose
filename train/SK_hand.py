import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import pickle


class SK_Model(object):
    def __init__(self, input_size, heatmap_size, batch_size, sk_index, stages=3, joints=21):
        self.joints = joints
        self.SK = np.array(sk_index)
        self.input_size = input_size
        self.heatmap_size = heatmap_size
        self.stages = stages
        self.batch_size = batch_size
        self.stage_heatmap = []
        self.stage_centermap=[]
        self.stage_loss = [0] * stages
        self.stage_center_loss=[0]*stages

        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=(None, self.input_size, self.input_size, 3),
                                                name='input_placeholder')

        self.heatmap_placeholder = tf.placeholder(dtype=tf.float32,
                                                  shape=(None,self.heatmap_size, self.heatmap_size, self.joints),
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

            for stage in range(1, self.stages + 1):
                center_map = self._get_center(stage)
                self._sk_layers(stage, center_map)

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
                mid_net = slim.conv2d(self.current_featuremap, 512, [3,3], scope='mid_net1')
                mid_net= slim.conv2d(mid_net, 128, [3, 3], scope='mid_net2')
                #mid_net = slim.conv2d(mid_net,128, [3, 3], scope='mid_net3')
                mid_net = slim.conv2d(mid_net, 128, [1, 1], scope='mid_net3')
                kps_0 = slim.conv2d(mid_net, 1, [1, 1], scope='key_points_0')
                self.stage_centermap.append(kps_0)
        return kps_0

    def _sk_layers(self, stage, center_map):

        with slim.arg_scope([slim.conv2d],
                            padding='SAME',
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):
            with tf.variable_scope('stage_' + str(stage), reuse=tf.AUTO_REUSE):
                reson_map = tf.concat([center_map, self.current_featuremap], axis=3)
                mid_net = slim.conv2d(reson_map, 128,  [3, 3], scope='midnet1_1')
                #mid_net = slim.conv2d(mid_net, 128, [3, 3], scope='midnet1_2')
                mid_net = slim.conv2d(mid_net, 128, [1, 1], scope='midnet1_2')
                key_points = slim.conv2d(mid_net, 5, [1, 1], scope='key_points_1')
                heatmap = tf.concat([center_map, key_points], axis=3)
                for i in range(1, 4):
                    reson_featuremap = tf.concat([key_points,
                                                  self.current_featuremap],
                                                 axis=3)
                    mid_net = slim.conv2d(reson_featuremap, 128, [3, 3], scope='midnet_' + str(i + 1) + '1')
                    #mid_net = slim.conv2d(mid_net, 128, [3, 3], scope='midnet_' + str(i + 1) + '2')
                    mid_net = slim.conv2d(mid_net, 128, [1, 1], scope='midnet_' + str(i + 1) + '2')
                    key_points = slim.conv2d(mid_net, 5, [1, 1], scope='key_points_' + str(i + 1))
                    heatmap = tf.concat([heatmap, key_points], axis=3)

            mid_map = heatmap[:, :, :, self.SK[0]][:, :, :, np.newaxis]
            for i in range(1, 21):
                mid_map = tf.concat([mid_map, heatmap[:, :, :, self.SK[i]][:, :, :, np.newaxis]], axis=3)
            self.stage_heatmap.append(mid_map)

    def build_loss(self, lr, lr_decay_rate, lr_decay_step, optimizer='Adam'):#RMSProp
        self.total_loss = 0
        self.learning_rate = lr
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_step = lr_decay_step

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.heatmap_placeholder,
                                                       name='l2_loss') / self.batch_size
            with tf.variable_scope('stage' + str(stage + 1) + 'center_loss'):
                self.stage_center_loss[stage] = tf.nn.l2_loss(self.stage_centermap[stage] - self.heatmap_placeholder[:,:,:,0][:,:,:,np.newaxis],
                                                       name='l2_loss')*5/ self.batch_size
            # tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]#self.stage_center_loss[stage]
            # tf.summary.scalar('total loss', self.total_loss)

        with tf.variable_scope('train'):
            self.global_step = tf.train.get_or_create_global_step()

            self.lr = tf.train.exponential_decay(self.learning_rate,
                                                 global_step=self.global_step,
                                                 decay_rate=self.lr_decay_rate,
                                                 decay_steps=self.lr_decay_step)
            # tf.summary.scalar('learning rate', self.lr)

            self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,
                                                            global_step=self.global_step,
                                                            learning_rate=self.lr,
                                                            optimizer=optimizer)
        # self.merged_summary = tf.summary.merge_all()

    def load_weights_from_file(self, weight_file_path, sess, finetune=True):
        # weight_file_object = open(weight_file_path, 'rb')
        weights = pickle.load(open(weight_file_path, 'rb'), encoding='latin1')

        with tf.variable_scope('', reuse=True):
            ## Pre stage conv
            # conv1
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/biases')

                loaded_kernel = weights['conv1_' + str(layer)]
                loaded_bias = weights['conv1_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv2
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/biases')

                loaded_kernel = weights['conv2_' + str(layer)]
                loaded_bias = weights['conv2_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv3
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/biases')

                loaded_kernel = weights['conv3_' + str(layer)]
                loaded_bias = weights['conv3_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv4
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/biases')

                loaded_kernel = weights['conv4_' + str(layer)]
                loaded_bias = weights['conv4_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 12) + '/weights')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 12) + '/biases')

                loaded_kernel = weights['conv5_' + str(layer)]
                loaded_bias = weights['conv5_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5_3_CPM
            conv_kernel = tf.get_variable('sub_stages/sub_stage_img_feature/weights')
            conv_bias = tf.get_variable('sub_stages/sub_stage_img_feature/biases')

            loaded_kernel = weights['conv5_3_CPM']
            loaded_bias = weights['conv5_3_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            if finetune != True:
                ## stage 1
                conv_kernel = tf.get_variable('stage_1/conv1/weights')
                conv_bias = tf.get_variable('stage_1/conv1/biases')

                loaded_kernel = weights['conv6_1_CPM']
                loaded_bias = weights['conv6_1_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

                conv_kernel = tf.get_variable('stage_1/stage_heatmap/weights')
                conv_bias = tf.get_variable('stage_1/stage_heatmap/biases')

                loaded_kernel = weights['conv6_2_CPM']
                loaded_bias = weights['conv6_2_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

                ## stage 2 and behind
                for stage in range(2, self.stages + 1):
                    for layer in range(1, 8):
                        conv_kernel = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/weights')
                        conv_bias = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/biases')

                        loaded_kernel = weights['Mconv' + str(layer) + '_stage' + str(stage)]
                        loaded_bias = weights['Mconv' + str(layer) + '_stage' + str(stage) + '_b']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))

