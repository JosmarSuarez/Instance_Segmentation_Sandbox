from menu_ui import *
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time
import os
from segm_threads import YolactThread, YolactArgs

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        self.yol_args = YolactArgs()

        MainWindow.setWindowTitle(self,"Model Visualizer")
        
        self.startButton.clicked.connect(self.start_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.stopButton.setEnabled(False)

        self.saveButton.clicked.connect(self.save_file)
        
        
        self.weights_path = "/home/josmar/proyectos/codes/03_model_visualizer/pyqt_window/weights"

        self.fill_comboBox(self.weightsBox, self.weights_path)
        self.set_weights()
        self.weightsBox.currentIndexChanged.connect(self.set_weights)
    

    def start_video(self):
        
        self.yol_args.only_mask = self.mask_radioButton.isChecked()
        

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
    
    def set_weights(self):
        weight = self.weightsBox.currentText()
        path = os.path.join(self.weights_path, weight)
        # print(self.yol_args.trained_model)
        if os.path.exists(path):
            self.yol_args.trained_model = path
        # print("After: ")
        # print(self.yol_args.trained_model)
    
    def fill_comboBox(self, box, path):
        if os.path.isdir:
            item_list = os.listdir(path)
        box.addItems(item_list)

        
    # def actualizar(self):
    #     self.label.setText("¡Acabas de hacer clic en el botón!")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()






