import tensorflow as tf

from SK_hand import SK_Model
from cpm_hand import CPM_Model
from config import SV
from operations import *


def main(argv):
    """

    :return:
    """
    """
    basic setting
    """
    pretrained_model_dir = os.path.join("train",SV.model_save_path, SV.pretrained_model_name)
    image_dir=os.path.join(SV.testdir,SV.testname)
    """
    load image
    """
    image=load_image(image_dir)
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

        #normalize the input picture
        img=image/255.0-0.5

        img=img[np.newaxis,:,:,:]

        heatmap = sess.run(sk.stage_heatmap[SV.stages - 1], feed_dict={sk.input_placeholder: img})

        lable=get_coods(heatmap)

        show_result(image,lable)




if __name__ == '__main__':
    tf.app.run()
