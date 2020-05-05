# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'GUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(600, 490)
        MainWindow.setMinimumSize(QtCore.QSize(600, 490))
        MainWindow.setMaximumSize(QtCore.QSize(600, 490))
        MainWindow.setStyleSheet("background-color: rgb(212, 212, 212);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.funlist = QtWidgets.QComboBox(self.centralwidget)
        self.funlist.setGeometry(QtCore.QRect(20, 70, 140, 40))
        self.funlist.setStyleSheet("font: 12pt \"Sans Serif\";\n"
"background-color: rgb(180, 180, 180);")
        self.funlist.setObjectName("funlist")
        self.funlist.addItem("")
        self.funlist.addItem("")
        self.funlist.addItem("")
        self.Txt = QtWidgets.QLabel(self.centralwidget)
        self.Txt.setGeometry(QtCore.QRect(20, 40, 70, 20))
        self.Txt.setStyleSheet("font: 12pt \"Sans Serif\";")
        self.Txt.setObjectName("Txt")
        self.runButton = QtWidgets.QPushButton(self.centralwidget)
        self.runButton.setGeometry(QtCore.QRect(32, 350, 120, 80))
        self.runButton.setStyleSheet("font: 20pt \"Sans Serif\";\n"
"background-color: rgb(180, 180, 180);")
        self.runButton.setObjectName("runButton")
        self.fileButton = QtWidgets.QPushButton(self.centralwidget)
        self.fileButton.setGeometry(QtCore.QRect(20, 210, 140, 40))
        self.fileButton.setStyleSheet("font: 12pt \"Sans Serif\";\n"
"background-color: rgb(180, 180, 180);")
        self.fileButton.setObjectName("fileButton")
        self.frame1 = QtWidgets.QFrame(self.centralwidget)
        self.frame1.setGeometry(QtCore.QRect(10, 30, 161, 421))
        self.frame1.setFrameShape(QtWidgets.QFrame.Box)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame1.setObjectName("frame1")
        self.frame2 = QtWidgets.QFrame(self.centralwidget)
        self.frame2.setGeometry(QtCore.QRect(190, 30, 391, 421))
        self.frame2.setFrameShape(QtWidgets.QFrame.Panel)
        self.frame2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame2.setObjectName("frame2")
        self.Imshow = QtWidgets.QLabel(self.centralwidget)
        self.Imshow.setGeometry(QtCore.QRect(200, 50, 368, 368))
        self.Imshow.setStyleSheet("")
        self.Imshow.setText("")
        self.Imshow.setObjectName("Imshow")
        self.frame2.raise_()
        self.frame1.raise_()
        self.funlist.raise_()
        self.Txt.raise_()
        self.runButton.raise_()
        self.fileButton.raise_()
        self.Imshow.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setStyleSheet("font: 16pt \"Sans Serif\"")
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Hand Pose Recognizer"))
        self.funlist.setItemText(0, _translate("MainWindow", "图片"))
        self.funlist.setItemText(1, _translate("MainWindow", "视频"))
        self.funlist.setItemText(2, _translate("MainWindow", "摄像头"))
        self.Txt.setText(_translate("MainWindow", "输入格式"))
        self.runButton.setText(_translate("MainWindow", "运行"))
        self.fileButton.setText(_translate("MainWindow", "选择文件"))
