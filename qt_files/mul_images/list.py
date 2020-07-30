from list_ui import *
import json
import random
import cv2
import json

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    dic={}
    files=[]
    is_useful = {}
    

    def __init__(self, *args, **kwargs):
        
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

               
        #self.listWidget.currentItemChanged.connect(self.clicked)
        
        self.dic, self.files = self.read_json('displaced.json')
        # self.list_files()
        self.init_useful_dic()
        self.model = QtGui.QStandardItemModel(self.listView)
        self.listView.setModel(self.model)
        self.list_files()
        self.model.itemChanged.connect(self.on_item_changed)
        self.selModel = self.listView.selectionModel()
        self.selModel.currentChanged.connect(self.clicked)
        self.saveButton.clicked.connect(self.save_list)
        
        
    def on_item_changed(self,item):
        current_key = item.index().data()
        self.is_useful[current_key] = bool(item.checkState())
        # self.is_useful[key] = item.checkState()

    def clicked(self, current, previous):
        
        item = current.data()
        key = item
        self.show_sample(key)
                
    def show_sample(self,key):
        self.clearLayout(self.gridLayout_2)
        sample = random.sample(self.dic[key], 9)
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
    
    # def list_files(self):
    #     #lists all the files listed inside the dictionary
    #     c=0
    #     for f in self.files:
    #         self.listView.insertItem(c, f)
    #         c+=1

    #lists all the files listed inside the dictionary
    def list_files(self):
        for f in self.files:
            # Create an item with a caption
            item = QtGui.QStandardItem(f)
        
            # Add a checkbox to it
            item.setCheckable(True)
        
            # Add the item to the model
            self.model.appendRow(item)
    
    def clearLayout(self,layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)

    def init_useful_dic(self):
        self.is_useful = {}
        for key in self.dic.keys():
            self.is_useful[key] = False
    
    def save_list(self):
        with open('is_useful.json', 'w') as f:
            json.dump(self.is_useful, f)
        msg = QtWidgets.QMessageBox()
        msg.setText("Lista guardada correctamente")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()
    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()