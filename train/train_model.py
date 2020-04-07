import tensorflow as tf
import numpy as np
import os
import glob

from SetValues import SV
from DS import DS
import model_units_funs
import data_funs
from cpm_hand import CPM_Model


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
    cpm = CPM_Model(SV.input_size,
                    SV.heatmap_size,
                    SV.batch_size,
                    SV.cpm_stage,
                    SV.joint)
    """build CPM model
    """
    cpm.build_model()
    cpm.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step)
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

                    cpm.load_weights_from_file(pretrained_model_dir, sess, finetune=True)

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

                heatmap=model_units_funs.generate_heatmap(SV.input_size,SV.heatmap_size,annotations)

                totol_loss, stage_loss, _, current_lr, \
                stage_heatmap_np, global_step = sess.run([cpm.total_loss,
                                                          cpm.stage_loss,
                                                          cpm.train_op,
                                                          cpm.lr,
                                                          cpm.stage_heatmap,
                                                          cpm.global_step],
                                                         feed_dict={cpm.input_placeholder: images,
                                                                    cpm.heatmap_placeholder: heatmap})
                if (turn+1)%10==0:
                    print("epsoid ", epsoid, ":")
                    print("totol loss is %f" % totol_loss)
                    for i in range(SV.cpm_stage):
                        print("stage%d loss: %f" % (i + 1, stage_loss[i]), end="  ")
                    print("")
        print("=====================train done==========================")
        saver.save(sess=sess, save_path=model_dir, global_step=(global_step + 1))
        print("\nModel checkpoint saved...\n")


if __name__ == '__main__':
    tf.app.run()
