from menu_ui import *
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import os
from yolact_files.segm_threads import YolactThread, YolactArgs
import re

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.yol_args = YolactArgs()

        MainWindow.setWindowTitle(self,"Model Visualizer")
        
        self.startButton.clicked.connect(self.start_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.stopButton.setEnabled(False)

        self.openButton.clicked.connect(self.open_file)
        self.saveButton.clicked.connect(self.save_file)
        
        self.weights_path = "/home/josmar/proyectos/codes/03_model_visualizer/pyqt_window/yolact_files/weights"

        # self.fill_comboBox(self.weightsBox, self.weights_path)
        
        self.fill_modelBox(self.modelBox, self.weights_path)
        
        self.set_model()
        self.set_weights()
        self.weightsBox.currentIndexChanged.connect(self.set_weights)
        self.modelBox.currentIndexChanged.connect(self.set_model)

        self.sizes = [[1920,1080], [1280,1024], [1280, 960], [1280,800], [1280,720], [1024, 768], [1024,576], [1024, 576],[960, 720], [864, 480],[800, 600],[800, 448], [640, 480], [640, 360], [432, 240], [352, 288], [320, 240], [176, 144], [160,120]]
        self.fill_resBox(self.resBox, self.sizes)

    def start_video(self):
        
        # self.set_model()
        actual_model = self.modelBox.currentText()
        self.yol_args.config = actual_model
        
        self.set_weights()
        self.yol_args.only_mask = not self.mask_checkBox.isChecked()
        self.yol_args.display_bboxes = self.bbox_checkBox.isChecked()
        self.yol_args.display_text = self.class_checkBox.isChecked()
        self.yol_args.display_scores = self.class_checkBox.isChecked()
        
        self.yol_args.size = self.resBox.currentText()
        
        if self.rb_webcam.isChecked():
            self.yol_args.video = str(self.webcam_id.value())
        if self.rb_file.isChecked():
            self.yol_args.video = self.line_in.text()

        self.video_thread = YolactThread(args = self.yol_args)
        self.video_thread.changePixmap.connect(self.setImage)
        self.video_thread.changeFPS.connect(self.showFPS)
        self.video_thread.start()

        self.label_video.setHidden(False)
        self.label_mask.setHidden(False)

        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)
        
        self.label_fps.setHidden(False)

    def stop_video(self):
        self.video_thread.running = False
        self.label_video.setHidden(True)
        self.label_mask.setHidden(True)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.yol_args = YolactArgs()
        self.label_fps.setText("")
        """
        setImage(self, image1, image2)
        
        Receives two QImages via pyqtSignal and shows them in the screen  
        """
    @pyqtSlot(QImage, QImage)
    def setImage(self, image1, image2):
        self.label_video.setPixmap(QPixmap.fromImage(image1))
        self.label_mask.setPixmap(QPixmap.fromImage(image2))
    
    @pyqtSlot(str)
    def showFPS(self, actual_fps):
        self.label_fps.setText(actual_fps)
    
    def save_file(self):
        _dir = QFileDialog.getSaveFileName(self, 'Save File')
        self.line_out.setText(_dir[0])

    def open_file(self):
        _dir = QFileDialog.getOpenFileName(self, 'Open File')
        self.line_in.setText(_dir[0])
    
    def set_weights(self):
        weight = self.weightsBox.currentText()
        path = os.path.join(self.weights_path, weight)
        # print(self.yol_args.trained_model)
        if os.path.exists(path):
            self.yol_args.trained_model = path
        # print("After: ")
        # print(self.yol_args.trained_model)

    def set_model(self):
        actual_model = self.modelBox.currentText()
        self.fill_weightBox(self.weightsBox, self.weights_path, actual_model)
        # self.yol_args.config = actual_model

    
    def fill_comboBox(self, box, path):
        if os.path.isdir:
            item_list = os.listdir(path)
        box.clear()
        box.addItems(item_list)
    
    def fill_weightBox(self,box,path,my_cfg):
        box.clear()
        my_cfg = my_cfg.replace("_config","")
        if os.path.isdir:
            item_list = os.listdir(path)
        weights_list=[item for item in item_list if item.find(my_cfg) != -1]
        box.clear()
        box.addItems(weights_list)

    def fill_modelBox(self,box,path):
        if os.path.isdir:
            item_list = os.listdir(path)
        cfg_list=[]
        for item in item_list:
            # print(item)
            results = [x.start() for x in re.finditer('\_', item)]
            _cfg = item[:results[-2]] + "_config"
            cfg_list.append(_cfg)
        box.clear()
        box.addItems(set(cfg_list))
    
    def fill_resBox(self, box, sizes):
        box.clear()
        str_size = [(str(size[0]) + "x" + str(size[1])) for size in sizes]
        box.addItems(str_size)
    # def actualizar(self):
    #     self.label.setText("¡Acabas de hacer clic en el botón!")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()






