from menu_ui import *
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import time


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, parent=None ):
        QtCore.QThread.__init__(self, parent)

    def run(self):
        cap = cv2.VideoCapture(0)
        # Check if camera opened successfully 
        if (cap.isOpened()== False):  
            print("Error opening video  file") 
        
        # Read until video is completed 
        while(cap.isOpened()): 
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            # Break the loop 
            else:  
                break

class TaskThread(QThread):
    # notifyProgress = pyqtSignal(int)
    changePixmap = pyqtSignal(QImage)
    def __init__(self, video_source, parent=None):
        QThread.__init__(self, parent)
        self.video_source = video_source
    def run(self):
        #use self.myvar in your run 
        # for i in range(self.myvar):
        #     self.notifyProgress.emit(i)
        #     time.sleep(0.1)
        self.running = True
        cap = cv2.VideoCapture(self.video_source)
        # Check if camera opened successfully 
        if (cap.isOpened()== False):  
            print("Error opening video  file") 
        
        # Read until video is completed 
        while(cap.isOpened() and self.running): 
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
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


    def start_video(self):
        self.video_thread = TaskThread(video_source="/home/josmar/Vídeos/krita_correction_ucb.mp4")
        self.video_thread.changePixmap.connect(self.setImage)
        self.video_thread.start()

    def stop_video(self):
        self.video_thread.running = False
        time.sleep(1)
        self.label_video.clear()
        self.label_mask.clear()
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label_video.setPixmap(QPixmap.fromImage(image))
        self.label_mask.setPixmap(QPixmap.fromImage(image))


        
    # def actualizar(self):
    #     self.label.setText("¡Acabas de hacer clic en el botón!")


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()






