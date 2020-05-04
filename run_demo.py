import tensorflow as tf

from SK_hand import SK_Model
from cpm_hand import CPM_Model
from config import SV
from operations import *
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

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
    #image=cv2.blur(image,(3,3))
    #image = cv2.GaussianBlur(image, (3,3), 0)
    #image=cv2.bilateralFilter(image,0,15,15)
    #image=cv2.medianBlur(image, 3)  poor effect
    #image=cv2.blur(image,(4,4))   poor effect
    #kernel=np.ones((3,3),np.uint8)
    #image = cv2.erode(image,kernel)
    #image = cv2.dilate(image, kernel)
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
    sk.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step, optimizer="RMSProp")#"RMSProp"
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

        """
        vedio oparetion
        """
        """kalman_array=kalman_init()
        cap = cv2.VideoCapture(0)
        while(cap.isOpened()):
            ret, frame = cap.read()  # frame shape=1080x1920
            frame = frame_resize(frame)
            img = cv2.GaussianBlur(frame, (9, 9), 1)
            img = img / 255.0 - 0.5

            img = img[np.newaxis, :, :, :]

            heatmap = sess.run(sk.stage_heatmap[SV.stages - 1], feed_dict={sk.input_placeholder: img})

            lable = get_coods(heatmap)
            lable=movement_adjust(lable,kalman_array)
            draw_skeleton(frame, lable)
            show_result(frame, lable,webcam=True)
            if cv2.waitKey(1)  == ord('q'):
                 break
        cap.release()
        cv2.destroyAllWindows()
        """
        cap = cv2.VideoCapture("./test/hand.mp4")
        kalman_array = kalman_init()
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        vw=cv2.VideoWriter('output.mp4',fourcc, 24.0, (368,368))
        while(cap.isOpened()):
            ret, frame = cap.read()  # frame shape=1080x1920
            if ret==False:
                break
            frame = frame_resize(frame)
            img = cv2.GaussianBlur(frame, (9, 9), 1)
            img = img / 255.0 - 0.5

            img = img[np.newaxis, :, :, :]

            heatmap = sess.run(sk.stage_heatmap[SV.stages - 1], feed_dict={sk.input_placeholder: img})

            lable = get_coods(heatmap)
            #lable = movement_adjust(lable, kalman_array)
            draw_skeleton(frame, lable)
            vw.write(frame)
            #show_result(frame, lable,webcam=True)
            #if cv2.waitKey(17) == 113:
                #break

        cap.release()
        vw.release()
        #cv2.destroyAllWindows()

        """#normalize the input picture
        img=image/255.0-0.5

        img=img[np.newaxis,:,:,:]

        heatmap = sess.run(sk.stage_heatmap[SV.stages - 1], feed_dict={sk.input_placeholder: img})

        lable=get_coods(heatmap)

        show_result(image,lable)

"""

if __name__ == '__main__':
    tf.app.run()
