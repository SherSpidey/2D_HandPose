from train.SK_hand import SK_Model
from old_ver.V1.SetValues import SV
from PIL import Image
from old_ver.V1.data_funs import *
from old_ver.V1.model_units_funs import *
from old_ver.V1.DS import DS
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main(argv):
    """

    :param argv:
    :return:
    """
    """model load path"""
    model_dir = os.path.join("train", SV.model_save_path, SV.model_name)

    """basic setting"""
    test_dir = os.path.join(SV.testdir, SV.testname)

    data = DS(SV.dataset_main_path,
              1,
              datasize(SV.mode),
              mode=SV.mode)
    """input resize"""
    file_assert(test_dir)
    image = Image.open(test_dir)
    if image.size[0] != SV.input_size or image.size[1] != SV.input_size:
        image = image.resize((SV.input_size, SV.input_size))
    image = np.array(image)
    image = image[np.newaxis, :]

    """load CPM model"""
    sk = SK_Model(SV.input_size,
                  SV.heatmap_size,
                  SV.batch_size,
                  SV.sk_index,
                  stages=SV.stages,
                  joints=SV.joint)
    """build CPM model
    """
    sk.build_model()
    sk.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step,optimizer="RMSProp")
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
            sk.load_weights_from_file(model_dir, sess, finetune=True)

            # Check weights
            for variable in tf.trainable_variables():
                with tf.variable_scope('', reuse=True):
                    var = tf.get_variable(variable.name.split(':0')[0])
                    print(variable.name, np.mean(sess.run(var)))

        else:
            saver.restore(sess, model_dir)
            print("load model done!")

        """ima,lab=data.NextBatch()
        heatmap = generate_heatmap(SV.input_size, SV.heatmap_size, lab)
        heatmap ,loss= sess.run([sk.stage_heatmap[SV.stages- 1],
                                sk.stage_loss[SV.stages- 1]],
                                feed_dict={sk.input_placeholder: ima,
                                           sk.heatmap_placeholder: heatmap})"""
        heatmap = sess.run(sk.stage_heatmap[SV.stages -1],
                                feed_dict={sk.input_placeholder: image})
        #print("heatmap:",heatmap[0])
        annotation = get_coods_v2(heatmap)
        #print("loss=",loss)
        reshow(image, annotation)
        #reshow(ima,lab)


if __name__ == '__main__':
    tf.app.run()
