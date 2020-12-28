from menu_ui import *
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import os
from centermask2_files.centermask_threads import CentermaskThread, CentermaskArgs
import re
# sys.path.append('/home/josmar/proyectos/centermask2')

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.centermask_args = CentermaskArgs()

        MainWindow.setWindowTitle(self,"Model Visualizer")
        
        self.startButton.clicked.connect(self.start_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.stopButton.setEnabled(False)

        self.saveButton.clicked.connect(self.save_file)
        
        
        self.weights_path = "/home/josmar/proyectos/codes/03_model_visualizer/pyqt_window/centermask2_files/weights"
        self.models_path = "/home/josmar/proyectos/codes/03_model_visualizer/pyqt_window/centermask2_files/configs/centermask"

        self.fill_comboBox(self.weightsBox, self.weights_path)
        self.fill_comboBox(self.modelBox, self.models_path)
        
        # self.fill_modelBox(self.modelBox, self.weights_path)
        

    

    def start_video(self):
        
        # self.set_model()
               
        
        
        self.centermask_args = CentermaskArgs()

        self.centermask_args.config_file = os.path.join(self.models_path, self.modelBox.currentText())
        
        weight = os.path.join(self.weights_path, self.weightsBox.currentText())
        if weight.find("run") == -1:
            self.centermask_args.opts = ["MODEL.WEIGHTS", weight]
        else:
            self.centermask_args.opts = ["MODEL.WEIGHTS", weight,
            "MODEL.FCOS.NUM_CLASSES", "1"]

        self.centermask_args.show_image = self.mask_checkBox.isChecked()
        self.centermask_args.show_boxes = self.bbox_checkBox.isChecked()
        self.centermask_args.show_labels = self.class_checkBox.isChecked()
        self.centermask_args.set_alpha = 1



        self.video_thread = CentermaskThread(args = self.centermask_args)
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
        self.video_thread.quit()


        self.label_video.setHidden(True)
        self.label_mask.setHidden(True)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
        self.centermask_args = CentermaskArgs()
        self.label_fps.setHidden(True)
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
        self.label_save.setText(_dir[0])


    
    def fill_comboBox(self, box, path):
        if os.path.isdir(path):
            item_list = os.listdir(path)
        item_list.sort()
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
        box.addItems(set(cfg_list))
    # def actualizar(self):
    #     self.label.setText("¡Acabas de hacer clic en el botón!")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()






