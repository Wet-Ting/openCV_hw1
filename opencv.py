# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'opencv.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(861, 474)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 30, 161, 231))
        self.groupBox.setObjectName("groupBox")
        self.loadImage = QtWidgets.QToolButton(self.groupBox)
        self.loadImage.setGeometry(QtCore.QRect(10, 30, 141, 41))
        self.loadImage.setObjectName("loadImage")
        self.colorConv = QtWidgets.QToolButton(self.groupBox)
        self.colorConv.setGeometry(QtCore.QRect(10, 80, 141, 41))
        self.colorConv.setObjectName("colorConv")
        self.blending = QtWidgets.QToolButton(self.groupBox)
        self.blending.setGeometry(QtCore.QRect(10, 180, 141, 41))
        self.blending.setObjectName("blending")
        self.imgFlipping = QtWidgets.QToolButton(self.groupBox)
        self.imgFlipping.setGeometry(QtCore.QRect(10, 130, 141, 41))
        self.imgFlipping.setObjectName("imgFlipping")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(190, 30, 151, 231))
        self.groupBox_3.setObjectName("groupBox_3")
        self.globalThr = QtWidgets.QToolButton(self.groupBox_3)
        self.globalThr.setGeometry(QtCore.QRect(10, 60, 131, 41))
        self.globalThr.setObjectName("globalThr")
        self.localThr = QtWidgets.QToolButton(self.groupBox_3)
        self.localThr.setGeometry(QtCore.QRect(10, 130, 131, 41))
        self.localThr.setObjectName("localThr")
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 270, 321, 111))
        self.groupBox_5.setObjectName("groupBox_5")
        self.gaussian = QtWidgets.QToolButton(self.groupBox_5)
        self.gaussian.setGeometry(QtCore.QRect(10, 20, 141, 31))
        self.gaussian.setObjectName("gaussian")
        self.sobelX = QtWidgets.QToolButton(self.groupBox_5)
        self.sobelX.setGeometry(QtCore.QRect(160, 20, 151, 31))
        self.sobelX.setObjectName("sobelX")
        self.sobelY = QtWidgets.QToolButton(self.groupBox_5)
        self.sobelY.setGeometry(QtCore.QRect(10, 70, 141, 31))
        self.sobelY.setObjectName("sobelY")
        self.magnitude = QtWidgets.QToolButton(self.groupBox_5)
        self.magnitude.setGeometry(QtCore.QRect(160, 70, 151, 31))
        self.magnitude.setObjectName("magnitude")
        self.buttonBox = QtWidgets.QDialogButtonBox(self.centralwidget)
        self.buttonBox.setGeometry(QtCore.QRect(670, 410, 161, 51))
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.groupBox_6 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_6.setGeometry(QtCore.QRect(350, 30, 291, 351))
        self.groupBox_6.setObjectName("groupBox_6")
        self.groupBox_7 = QtWidgets.QGroupBox(self.groupBox_6)
        self.groupBox_7.setGeometry(QtCore.QRect(20, 30, 251, 261))
        self.groupBox_7.setObjectName("groupBox_7")
        self.groupBox_8 = QtWidgets.QGroupBox(self.groupBox_7)
        self.groupBox_8.setGeometry(QtCore.QRect(10, 20, 231, 181))
        self.groupBox_8.setObjectName("groupBox_8")
        self.angle = QtWidgets.QTextEdit(self.groupBox_8)
        self.angle.setGeometry(QtCore.QRect(70, 20, 81, 31))
        self.angle.setObjectName("angle")
        self.scale = QtWidgets.QTextEdit(self.groupBox_8)
        self.scale.setGeometry(QtCore.QRect(70, 60, 81, 31))
        self.scale.setObjectName("scale")
        self.tx = QtWidgets.QTextEdit(self.groupBox_8)
        self.tx.setGeometry(QtCore.QRect(70, 100, 81, 31))
        self.tx.setObjectName("tx")
        self.ty = QtWidgets.QTextEdit(self.groupBox_8)
        self.ty.setGeometry(QtCore.QRect(70, 140, 81, 31))
        self.ty.setObjectName("ty")
        self.label_3 = QtWidgets.QLabel(self.groupBox_8)
        self.label_3.setGeometry(QtCore.QRect(20, 30, 41, 21))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox_8)
        self.label_4.setGeometry(QtCore.QRect(20, 70, 41, 21))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox_8)
        self.label_5.setGeometry(QtCore.QRect(20, 110, 41, 21))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox_8)
        self.label_6.setGeometry(QtCore.QRect(20, 150, 41, 21))
        self.label_6.setObjectName("label_6")
        self.label_11 = QtWidgets.QLabel(self.groupBox_8)
        self.label_11.setGeometry(QtCore.QRect(170, 30, 41, 21))
        self.label_11.setObjectName("label_11")
        self.label_12 = QtWidgets.QLabel(self.groupBox_8)
        self.label_12.setGeometry(QtCore.QRect(170, 110, 41, 21))
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.groupBox_8)
        self.label_13.setGeometry(QtCore.QRect(170, 150, 41, 21))
        self.label_13.setObjectName("label_13")
        self.RST = QtWidgets.QToolButton(self.groupBox_7)
        self.RST.setGeometry(QtCore.QRect(10, 210, 231, 41))
        self.RST.setObjectName("RST")
        self.PT = QtWidgets.QToolButton(self.groupBox_6)
        self.PT.setGeometry(QtCore.QRect(30, 300, 231, 41))
        self.PT.setObjectName("PT")
        self.groupBox_9 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_9.setGeometry(QtCore.QRect(660, 30, 181, 351))
        self.groupBox_9.setObjectName("groupBox_9")
        self.Imgs = QtWidgets.QToolButton(self.groupBox_9)
        self.Imgs.setGeometry(QtCore.QRect(10, 30, 151, 41))
        self.Imgs.setObjectName("Imgs")
        self.Hyper = QtWidgets.QToolButton(self.groupBox_9)
        self.Hyper.setGeometry(QtCore.QRect(10, 80, 151, 41))
        self.Hyper.setObjectName("Hyper")
        self.Train = QtWidgets.QToolButton(self.groupBox_9)
        self.Train.setGeometry(QtCore.QRect(10, 130, 151, 41))
        self.Train.setObjectName("Train")
        self.Result = QtWidgets.QToolButton(self.groupBox_9)
        self.Result.setGeometry(QtCore.QRect(10, 180, 151, 41))
        self.Result.setObjectName("Result")
        self.Inf = QtWidgets.QToolButton(self.groupBox_9)
        self.Inf.setGeometry(QtCore.QRect(10, 290, 151, 41))
        self.Inf.setObjectName("Inf")
        self.label_14 = QtWidgets.QLabel(self.groupBox_9)
        self.label_14.setGeometry(QtCore.QRect(10, 230, 91, 21))
        self.label_14.setObjectName("label_14")
        self.Test_idx = QtWidgets.QTextEdit(self.groupBox_9)
        self.Test_idx.setGeometry(QtCore.QRect(10, 250, 151, 31))
        self.Test_idx.setObjectName("Test_idx")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Image Processing"))
        self.loadImage.setText(_translate("MainWindow", "1.1 Load Image"))
        self.colorConv.setText(_translate("MainWindow", "1.2 Color Conversion"))
        self.blending.setText(_translate("MainWindow", "1.4 Blending"))
        self.imgFlipping.setText(_translate("MainWindow", "1.3. Image Flipping"))
        self.groupBox_3.setTitle(_translate("MainWindow", "2. Adaptive Threshold"))
        self.globalThr.setText(_translate("MainWindow", "2.1 Global Threshold"))
        self.localThr.setText(_translate("MainWindow", "2.2 Local Threshold"))
        self.groupBox_5.setTitle(_translate("MainWindow", "4. Convolution"))
        self.gaussian.setText(_translate("MainWindow", "4.1 Gaussian"))
        self.sobelX.setText(_translate("MainWindow", "4.2 Sobel X"))
        self.sobelY.setText(_translate("MainWindow", "4.3 Sobel Y"))
        self.magnitude.setText(_translate("MainWindow", "4.4 Magnitude"))
        self.groupBox_6.setTitle(_translate("MainWindow", "3. Image Transformation"))
        self.groupBox_7.setTitle(_translate("MainWindow", "3.1 Rot, scale, Translate"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Parameters"))
        self.angle.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))
        self.scale.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1</p></body></html>"))
        self.tx.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))
        self.ty.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">0</p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Angle:</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Scale:</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Tx:</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Ty:</span></p></body></html>"))
        self.label_11.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">deg</span></p></body></html>"))
        self.label_12.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">pixel</span></p></body></html>"))
        self.label_13.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">pixel</span></p></body></html>"))
        self.RST.setText(_translate("MainWindow", "3.1 Rotation, scaling, translation"))
        self.PT.setText(_translate("MainWindow", "3.2 Perspective Transform"))
        self.groupBox_9.setTitle(_translate("MainWindow", "Train Cifar-10 Classifier"))
        self.Imgs.setText(_translate("MainWindow", "5.1 Show Train Images"))
        self.Hyper.setText(_translate("MainWindow", "5.2 Show Hyperparameters"))
        self.Train.setText(_translate("MainWindow", "5.3 Train 1 Epoch"))
        self.Result.setText(_translate("MainWindow", "5.4 Show Training Result"))
        self.Inf.setText(_translate("MainWindow", "5.5 Inference"))
        self.label_14.setText(_translate("MainWindow", "<html><head/><body><p>Test Image Index:</p></body></html>"))
        self.Test_idx.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'PMingLiU\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"right\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">(0~9999)</p></body></html>"))