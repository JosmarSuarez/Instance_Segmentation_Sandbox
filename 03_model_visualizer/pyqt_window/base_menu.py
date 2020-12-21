from menu_ui import *
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication, QFileDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time




class TaskThread(QThread):
    # notifyProgress = pyqtSignal(int)
    changePixmap = pyqtSignal(QImage, QImage)
    def __init__(self, video_source, parent=None):
        QThread.__init__(self, parent)
        self.video_source = video_source
    
    def set_res(self, cap, x,y):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1) 
    
    def convert_to_qt(self, rgbImage):
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
        p = convertToQtFormat.scaled(800, 800, Qt.KeepAspectRatio)
        return p

    def run(self):
        #use self.myvar in your run 
        # for i in range(self.myvar):
        #     self.notifyProgress.emit(i)
        #     time.sleep(0.1)
        self.running = True
        cap = cv2.VideoCapture(self.video_source)
        self.set_res(cap, 1280, 720)

        # Check if camera opened successfully 
        if (cap.isOpened()== False):  
            print("Error opening video  file") 
        
        # Read until video is completed 
        while(cap.isOpened() and self.running): 
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                # rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rotatedImage = cv2.flip(rgbImage, 1)

                qt_original = self.convert_to_qt(rgbImage)
                qt_rotated = self.convert_to_qt(rotatedImage)
                self.changePixmap.emit(qt_original, qt_rotated)

            # Break the loop 
            else:  
                break
        



class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        MainWindow.setWindowTitle(self,"Model Visualizer")
        
        self.startButton.clicked.connect(self.start_video)
        self.stopButton.clicked.connect(self.stop_video)
        self.stopButton.setEnabled(False)

        self.saveButton.clicked.connect(self.save_file)
    

    def start_video(self):
        
        self.video_thread = TaskThread(video_source=1)
        self.video_thread.changePixmap.connect(self.setImage)
        self.video_thread.start()

        self.label_video.setHidden(False)
        self.label_mask.setHidden(False)

        self.startButton.setEnabled(False)
        self.stopButton.setEnabled(True)

    def stop_video(self):
        self.video_thread.running = False
        self.label_video.setHidden(True)
        self.label_mask.setHidden(True)
        self.startButton.setEnabled(True)
        self.stopButton.setEnabled(False)
    
        """
        setImage(self, image1, image2)
        
        Receives two QImages via pyqtSignal and shows them in the screen  
        """
    @pyqtSlot(QImage, QImage)
    def setImage(self, image1, image2):
        self.label_video.setPixmap(QPixmap.fromImage(image1))
        self.label_mask.setPixmap(QPixmap.fromImage(image2))
    
    def save_file(self):
        _dir = QFileDialog.getSaveFileName(self, 'Save File')
        self.label_save.setText(_dir[0])
        
    # def actualizar(self):
    #     self.label.setText("¡Acabas de hacer clic en el botón!")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()






