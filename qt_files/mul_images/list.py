from list_ui import *
import json
import random
import cv2

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

               
        self.listWidget.currentItemChanged.connect(self.clicked)
        global dic
        dic, files = self.read_json('displaced.json')
        

        #lists all the files listed inside the dictionary
        c=0
        for f in files:
            self.listWidget.insertItem(c, f)
            c+=1
        
    
    def clicked(self, qmodelindex):
        item = self.listWidget.currentItem()
        key = item.text()
                
        self.clearLayout(self.gridLayout_2)
        sample = random.sample(dic[key], 9)
        labels=[]
        c=0

        for img_name in sample:
            i_label = QtWidgets.QLabel()
            # i_label.setText(img_name)

            sil_path = "../../datasets/casia_B1_silhouettes/{}-{}.png".format(key, img_name)
            img_path = "../../datasets/casia_B1_images/{}-{}.jpg".format(key, img_name)
            img = cv2.imread(img_path)
            sil = cv2.imread(sil_path)

            alpha = 0.4
            beta = (1.0-alpha)
            new_img = cv2.addWeighted(img, alpha, sil, beta, 0.0)
            #cv2 image converted to qt image
            qtimg = QtGui.QImage(new_img.data, img.shape[1], img.shape[0],QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(qtimg))
            i_label.setPixmap(pixmap)
            labels.append(i_label)

            self.gridLayout_2.addWidget(i_label, c//3, c%3, 1, 1)
            c+=1
    
    def read_json(self,path):
        with open(path) as f:
            folders = json.load(f)
            list_folders = list(folders.keys())
            list_folders.sort() 
        return  folders, list_folders

    
    def clearLayout(self,layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)
    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()