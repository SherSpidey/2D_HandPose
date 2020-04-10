from train.KRT_model import KRT
from train.SetValues import SV
from PIL import Image
from train.data_funs import *
from train.model_units_funs import *


def main(argv):
    """

    :param argv:
    :return:
    """
    """model load path"""
    model_dir = os.path.join("train", SV.model_save_path, SV.pretrained_model_name)

    """basic setting"""
    test_dir = os.path.join(SV.testdir, SV.testname)

    """input resize"""
    file_assert(test_dir)
    image = Image.open(test_dir)
    if image.size[0] != SV.input_size or image.size[1] != SV.input_size:
        image = image.resize((SV.input_size, SV.input_size))
    image = np.array(image)
    image = image[np.newaxis, :]

    """load CPM model"""
    krt = KRT(SV.input_size,
                    SV.heatmap_size,
                    SV.batch_size,
                    SV.sk_index,
                    SV.joint)
    """build CPM model
    """
    krt.build_model()
    krt.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step,optimizer="Adam")
    print('\n=====Model Build=====\n')

    with tf.Session() as sess:
        # Create model saver
        saver = tf.train.Saver(max_to_keep=None)

        # Init all vars
        init = tf.global_variables_initializer()
        sess.run(init)

        # Restore pretrained weights
        print("Now loading model!")
        if model_dir.endswith('.pkl'):
            #krt.load_weights_from_file(model_dir, sess, finetune=True)

            # Check weights
            for variable in tf.trainable_variables():
                with tf.variable_scope('', reuse=True):
                    var = tf.get_variable(variable.name.split(':0')[0])
                    print(variable.name, np.mean(sess.run(var)))

        else:
            saver.restore(sess, model_dir)
            print("load model done!")

        heatmap = sess.run(krt.heatmap, feed_dict={krt.input_placeholder: image})

        annotation = get_coords_from_heatmap(heatmap)

        reshow(image, annotation)


if __name__ == '__main__':
    tf.app.run()
