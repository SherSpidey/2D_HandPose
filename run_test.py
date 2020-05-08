import tensorflow as tf

from train.SK_hand import SK_Model
from train.cpm_hand import CPM_Model
from train.config import SV
from data_model import DS
from train.operations import *


def main(argv):
    """

    :return:
    """
    """
    basic setting
    """
    pretrained_model_dir = os.path.join("train", SV.model_save_path, SV.pretrained_model_name)
    l2_loss = 0
    """
    load dataset 
    """

    data = DS(os.path.join("train", SV.dataset_main_path),
              SV.batch_size,
              mode=SV.mode)

    """
    load CPM model
    """
    if SV.model == "cpm_sk":
        sk = SK_Model(SV.input_size,
                      SV.heatmap_size,
                      SV.batch_size,
                      SV.sk_index,
                      stages=SV.stages,
                      joints=SV.joint)
    else:
        sk = CPM_Model(SV.input_size,
                       SV.heatmap_size,
                       SV.batch_size,
                       stages=SV.stages,
                       joints=SV.joint + 1)
    """
    build CPM model
    """
    sk.build_model()
    sk.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step, optimizer="RMSProp")  # "RMSProp"
    print('\n=====Model Build=====\n')

    with tf.Session() as sess:

        # Create model saver
        saver = tf.train.Saver(max_to_keep=None)

        # Init all vars
        init = tf.global_variables_initializer()
        sess.run(init)
        # Restore pretrained weights
        if SV.pretrained_model_name != "":
            print("Now loading model!")
            if SV.pretrained_model_name.endswith('.pkl'):
                if SV.model == "cpm_sk":
                    sk.load_weights_from_file(pretrained_model_dir, sess, finetune=True)
                else:
                    sk.load_weights_from_file(pretrained_model_dir, sess, finetune=False)
                print("load model done!")

                # Check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))

            else:
                saver.restore(sess, pretrained_model_dir)
                print("load model done!")

                # check weights
                for variable in tf.trainable_variables():
                    with tf.variable_scope('', reuse=True):
                        var = tf.get_variable(variable.name.split(':0')[0])
                        print(variable.name, np.mean(sess.run(var)))

        for i in range (3680//2):
            img, ano = data.NextBatch()
            img = img / 255.0 - 0.5

            heatmap = sess.run(sk.stage_heatmap[SV.stages - 1], feed_dict={sk.input_placeholder: img})

            lable = get_coods(heatmap,train=True)

            l2_loss += np.linalg.norm(lable - ano) / SV.batch_size

            print("%d of 3680."%((i+1)*SV.batch_size))

        l2_loss = l2_loss / 3680

        print("L2 loss for evaluation is ", l2_loss)


if __name__ == '__main__':
    tf.app.run()
