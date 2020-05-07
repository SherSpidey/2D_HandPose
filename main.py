import sys
import time
import qimage2ndarray
import tensorflow as tf
from train.SK_hand import SK_Model
from train.cpm_hand import CPM_Model
from train.config import SV
from train.operations import *
from GUI import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPixmap


class RunGUI(QMainWindow, Ui_MainWindow):
    def __del__(self):
        try:
            self.camera.release()  # 释放资源
            self.sess.close()
        except:
            return

    def __init__(self, parent=None):
        super(RunGUI, self).__init__(parent)
        self.filename = None
        self.Model_loaded = False
        self.Image = []
        self.Image_v2 = []
        self.choice = "图片"
        self.kalman_array = kalman_init()
        self.setupUi(self)
        self.CallBackFunctions()
        self.Timer = QTimer()
        self.Timer.timeout.connect(self.TimerOutFun)
        self.Model_init()

    def Model_init(self):
        model_dir = os.path.join("train", SV.model_save_path, SV.pretrained_model_name)
        """
            load CPM model
            """
        if SV.model == "cpm_sk":
            self.Model = SK_Model(SV.input_size,
                                  SV.heatmap_size,
                                  SV.batch_size,
                                  SV.sk_index,
                                  stages=SV.stages,
                                  joints=SV.joint)
        else:
            self.Model = CPM_Model(SV.input_size,
                                   SV.heatmap_size,
                                   SV.batch_size,
                                   stages=SV.stages,
                                   joints=SV.joint + 1)
        """
        build CPM model
        """
        self.Model.build_model()
        self.Model.build_loss(SV.learning_rate, SV.lr_decay_rate, SV.lr_decay_step, optimizer="RMSProp")

        self.sess = tf.Session()
        # Create model saver
        saver = tf.train.Saver(max_to_keep=None)

        # Init all vars
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Restore pretrained weights
        saver.restore(self.sess, model_dir)
        # Check weights
        for variable in tf.trainable_variables():
            with tf.variable_scope('', reuse=True):
                var = tf.get_variable(variable.name.split(':0')[0])
                print(variable.name, np.mean(self.sess.run(var)))

    def run(self):
        if self.choice != "摄像头" and (len(self.Image) == 0 or len(self.Image_v2)) == 0:
            pass
        else:
            if self.runButton.text() == "运行":
                if self.choice == "图片":

                    img = self.Image_v2 / 255.0 - 0.5

                    img = img[np.newaxis, :, :, :]

                    heatmap = self.sess.run(self.Model.stage_heatmap[SV.stages - 1],
                                            feed_dict={self.Model.input_placeholder: img})

                    lable = get_coods(heatmap)
                    draw_skeleton(self.Image, lable)
                    self.im_display(self.Image)
                    self.statusbar.showMessage("识别完成！")

                elif self.choice == "视频":
                    self.runButton.setText("暂停")
                    self.cap = cv2.VideoCapture(self.filename)
                    self.Timer.start(1)
                    self.timelb = time.clock()
                else:
                    self.runButton.setText("暂停")
                    self.cap = cv2.VideoCapture(0)
                    self.Timer.start(1)
                    self.timelb = time.clock()
            else:
                self.cap.release()
                self.display_clear()
                self.runButton.setText("运行")
                self.statusbar.showMessage("")

    def CallBackFunctions(self):
        self.funlist.currentTextChanged.connect(self.getchoice)
        self.fileButton.clicked.connect(self.openfile)
        self.runButton.clicked.connect(self.run)

    def getchoice(self):
        self.choice = self.funlist.currentText()

    def openfile(self):
        if self.choice == "图片":
            self.filename, _ = QFileDialog.getOpenFileName(None, '选择文件', '', 'Image Files(*.jpg *.png);;All Files(*.*)')
            self.statusbar.showMessage('图片已加载')
            self.im_display()
        elif self.choice == "视频":
            self.filename, _ = QFileDialog.getOpenFileName(None, '选择文件', '', 'Video Files(*.mp4 *.avi);;All Files(*.*)')
        else:
            QMessageBox.warning(self, "警告", "当前设置为摄像头！")

    def im_display(self, pre_img=[]):
        if len(pre_img) == 0:
            self.Image = load_image(self.filename)
            self.Image = frame_resize(self.Image)
            self.Image_v2 = cv2.GaussianBlur(self.Image, (9, 9), 1)
        img = cv2.cvtColor(self.Image, cv2.COLOR_BGR2RGB)
        qimg = qimage2ndarray.array2qimage(img)
        self.Imshow.setPixmap(QPixmap(qimg))
        self.Imshow.show()

    def display_clear(self):
        img = np.ones((368, 368, 3), dtype="uint8") * 180
        qimg = qimage2ndarray.array2qimage(img)
        self.Imshow.setPixmap(QPixmap(qimg))
        self.Imshow.show()

    def TimerOutFun(self):
        success, frame = self.cap.read()
        if success:
            frame = cv2.flip(frame, 1)  # 镜像翻转
            frame = frame_resize(frame)
            self.Image = frame
            img = cv2.GaussianBlur(frame, (9, 9), 1)
            img = img / 255.0 - 0.5

            img = img[np.newaxis, :, :, :]

            heatmap = self.sess.run(self.Model.stage_heatmap[SV.stages - 1],
                                    feed_dict={self.Model.input_placeholder: img})

            lable = get_coods(heatmap)
            lable = movement_adjust(lable, self.kalman_array, enable=False)
            draw_skeleton(self.Image, lable)
            self.im_display(self.Image)
            self.statusbar.showMessage("正在显示...")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = RunGUI()
    ui.show()
    sys.exit(app.exec_())
