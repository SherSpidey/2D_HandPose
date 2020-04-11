import tensorflow as tf
import numpy as np
import os
import glob

from SetValues import SV
from DS import DS
import model_units_funs
import data_funs
from SK_hand import SK_Model


def main(argv):
    """

    :param argv:
    :return:
    """
    """basic setting
    """
    model_dir = os.path.join(SV.model_save_path, SV.model_name)
    pretrained_model_dir = os.path.join(SV.model_save_path, SV.pretrained_model_name+'*')

    """load dataset and annotation
    """
    data = DS(SV.dataset_main_path,
              SV.batch_size,
              data_funs.datasize(SV.mode),
              mode=SV.mode)

    """load CPM model
    """
    sk = SK_Model(SV.input_size,
                  SV.heatmap_size,
                  SV.batch_size,
                  SV.sk_index,
                  stages=SV.stages,
                  joints=SV.joint)
    """build CPM model
    """
    sk.build_model()
    sk.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step,optimizer="Adam")
    print('\n=====Model Build=====\n')

    """training
    """
    with tf.Session() as sess:
        # Create model saver
        saver = tf.train.Saver(max_to_keep=None)

        # Init all vars
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore pretrained weights
        if SV.pretrained_model_name != "":
            if len(glob.glob(pretrained_model_dir))!=0:
                print("Now loading model!")
                if SV.pretrained_model_name.endswith('.pkl'):
                    sk.load_weights_from_file(pretrained_model_dir, sess, finetune=True)

                    # Check weights
                    for variable in tf.trainable_variables():
                        with tf.variable_scope('', reuse=True):
                            var = tf.get_variable(variable.name.split(':0')[0])
                            print(variable.name, np.mean(sess.run(var)))

                else:
                    saver.restore(sess, model_dir)
                    print("load model done!")

                    # check weights
                    for variable in tf.trainable_variables():
                        with tf.variable_scope('', reuse=True):
                            var = tf.get_variable(variable.name.split(':0')[0])
                            print(variable.name, np.mean(sess.run(var)))

        print("\n============training=================\n")
        for epsoid in range(SV.episodes):
            # Forward and update weights
            for turn in range(SV.epo_turns):
                images, annotations = data.NextBatch()
                #get all stage's heatmap
                #"""
                heatmap = []
                variance = np.arange(sk.stages, 0, -1)
                variance = np.sqrt(variance)
                for i in range(sk.stages):
                    heatmap.append(model_units_funs.generate_heatmap(SV.input_size,
                                                                     SV.heatmap_size, annotations, variance[i]))
                heatmap = np.array(heatmap)
                heatmap = np.transpose(heatmap, (1, 0, 2, 3, 4))
                #"""
                #heatmap=model_units_funs.generate_heatmap(SV.input_size,SV.heatmap_size, annotations)

                totol_loss, stage_loss, _, current_lr, \
                stage_heatmap_np, global_step = sess.run([sk.total_loss,
                                                          sk.stage_loss,
                                                          sk.train_op,
                                                          sk.lr,
                                                          sk.stage_heatmap,
                                                          sk.global_step],
                                                          feed_dict={sk.input_placeholder: images,
                                                                    sk.heatmap_placeholder: heatmap})
                if (turn+1)%10==0:
                    print("epsoid ", epsoid, ":")
                    print("learning rate: ",current_lr)
                    print("totol loss is %f" % totol_loss)
                    for i in range(SV.stages):
                        print("stage%d loss: %f" % (i + 1, stage_loss[i]), end="  ")
                    print("")
            if (epsoid+1)%3==0:
                saver.save(sess=sess, save_path=model_dir, global_step=(global_step + 1))
                print("\nModel checkpoint saved...\n")
        print("=====================train done==========================")



if __name__ == '__main__':
    tf.app.run()
