# py38_cvdl

from UI import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore 
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtGui import QImage, QPixmap,QFont
from matplotlib import pyplot as plt
import os
import cv2
import numpy as np
import vgg19 

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,parent=None):
        super(MainWindow, self).__init__(parent) # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()
        self.img = None
        self.imglabel = None

    def setup_control(self):
        self.ui.LoadImg.clicked.connect(self.load_img_click)
        self.ui.ShowTrainImages.clicked.connect(self.show_train_images_click)
        self.ui.ShowModelStructure.clicked.connect(self.show_model_structure_click)
        self.ui.ShowDataAugmentation.clicked.connect(self.show_data_augmentation_click)
        self.ui.ShowAccuracyAndLoss.clicked.connect(self.show_accuracy_and_loss_click)
        self.ui.Inference.clicked.connect(self.inference_click)




    def load_img_click(self):
        self.img, self.imglabel = vgg19.load_image_ui()
        print("Label: ",self.imglabel)

        test_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        test_image = cv2.resize(test_image, (256, 256), interpolation=cv2.INTER_AREA)
        height, width, channel = test_image.shape
        bytesPerline = 3 * width
        self.qImg = QImage(test_image.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.ImgBox.setPixmap(QPixmap.fromImage(self.qImg))


    def show_train_images_click(self):
        vgg19.show_training_img()

    def show_model_structure_click(self):
        vgg19.print_model()

    def show_data_augmentation_click(self):
        vgg19.Show_Data_Augmentation(self.img)

    def show_accuracy_and_loss_click(self):
        pixmap = QPixmap("AccuracyLoss.png")
        self.ui.ImgBox.setPixmap(pixmap)  
        self.ui.ImgBox.setScaledContents(True) 

    def inference_click(self):
        label_map = {0: 'airplane',  1: 'automobile',  2: 'bird',  3: 'cat',
                    4: 'deer',  5: 'dog',  6: 'frog',  7: 'horse',  8: 'ship',  9: 'truck'}
        result,resultclass=vgg19.test(self.img)
        result_text = "Confidence = " + str(result) + "\nPrediciton Label : "+ str(label_map[resultclass[0]])
        self.ui.ShowText.setText(result_text)



if __name__=='__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
