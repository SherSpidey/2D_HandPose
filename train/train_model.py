import tensorflow as tf

from SK_hand import SK_Model
from cpm_hand import CPM_Model
from config import SV
from data_model import DS
from operations import *


def main(argv):
    """

    :return:
    """
    """
    basic setting
    """
    model_dir = os.path.join(SV.model_save_path, SV.model_name)
    pretrained_model_dir = os.path.join(SV.model_save_path, SV.pretrained_model_name)

    """
    load dataset 
    """
    data = DS(SV.dataset_main_path,
              SV.batch_size,
              mode=SV.mode)

    """
    load CPM model
    """
    if SV.model=="cpm_sk":
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
                      joints=SV.joint+1)
    """
    build CPM model
    """
    sk.build_model()
    sk.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step, optimizer="Adam")#"RMSProp"
    print('\n=====Model Build=====\n')

    """
    training
    """
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
                if SV.model=="cpm_sk":
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

        print("\n============training=================\n")
        for epsoid in range(SV.episodes):
            # Forward and update weights
            for turn in range(SV.epo_turns):
                images, annotations = data.NextBatch()
                #normalize the input picture
                images=images/255.0-0.5
                #get heatmap
                heatmap = generate_heatmap(SV.input_size, SV.heatmap_size, annotations,model=SV.model)

                if SV.model=="cpm_sk":
                    totol_loss, stage_loss, _, current_lr, center_loss, \
                    stage_heatmap_np, global_step = sess.run([sk.total_loss,
                                                              sk.stage_loss,
                                                              sk.train_op,
                                                              sk.lr,
                                                              sk.stage_center_loss,
                                                              sk.stage_heatmap,
                                                              sk.global_step],
                                                              feed_dict={sk.input_placeholder: images,
                                                                         sk.heatmap_placeholder: heatmap})
                else:
                    totol_loss, stage_loss, _, current_lr, \
                    stage_heatmap_np, global_step = sess.run([sk.total_loss,
                                                              sk.stage_loss,
                                                              sk.train_op,
                                                              sk.lr,
                                                              sk.stage_heatmap,
                                                              sk.global_step],
                                                              feed_dict={sk.input_placeholder: images,
                                                                        sk.heatmap_placeholder: heatmap})
                if (turn + 1) % 10 == 0:
                    print("epsoid ", epsoid, ":")
                    print("learning rate: ", current_lr)
                    print("totol loss is %f" % totol_loss)
                    for i in range(SV.stages):
                        print("stage%d loss: %f" % (i + 1, stage_loss[i]), end="  ")
                    print("")
                    if SV.model=="cpm_sk":
                        for i in range(SV.stages):
                            print("center%d loss: %f" % (i + 1, center_loss[i]), end="  ")
                        print("")

            if (epsoid + 1) % 3 == 0:
                saver.save(sess=sess, save_path=model_dir, global_step=(global_step + 1))
                print("\nModel checkpoint saved...\n")
            print("=====================train done==========================")


if __name__ == '__main__':
    tf.app.run()
