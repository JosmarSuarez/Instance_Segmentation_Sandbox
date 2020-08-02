from list_ui import *
import json
import random
import cv2
import json
import os

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    dic={}
    files=[]
    is_useful = {}
    alpha = 0.4
    sli_min,sli_max = 0, 10

    key=""
    sample = []

    com_lists = []

    curr_box = ["001","nm"]

    check_path = ""
    

    def __init__(self, *args, **kwargs):
        
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)

        MainWindow.setWindowTitle(self,"Mask Viewer")       
        #self.listWidget.currentItemChanged.connect(self.clicked)
        
        self.dic = self.read_json('folder_list_cat.json')
        self.update_lists()
        self.update_combo()

        self.subBox.currentIndexChanged.connect(self.update_curr_box)
        self.typeBox.currentIndexChanged.connect(self.update_curr_box)

        self.model = QtGui.QStandardItemModel(self.listView)
        self.listView.setModel(self.model)

        self.saveButton.clicked.connect(self.save_list)
        
        self.alphaSlider.valueChanged.connect(self.on_slider_changed)
        self.alphaSlider.setValue(self.alpha*self.sli_max)
        self.alphaLabel.setText(str(self.alpha))

        self.randomButton.clicked.connect(self.randomize)
        
    def val2alpha(self,val):
        return val/self.sli_max

    def on_slider_changed(self,value):
        self.alpha = self.val2alpha(value)
        self.alphaLabel.setText(str(self.alpha))
        self.show_sample()

    def on_check_changed(self,item):
        current_key = item.index().data()
        print(current_key)
        self.is_useful[current_key] = item.checkState()
        # self.is_useful[key] = item.checkState()
        

    def on_list_changed(self, current, previous):
        if(current.data()!=None):
            self.key = current.data()
            self.create_sample()
            self.show_sample()
                
    def create_sample(self, sample_size=9):
        keys = self.key.split("-")
        num_files =len(self.dic[keys[0]][keys[1]][keys[2]][keys[3]]) 
        if num_files <9:
            sample_size = num_files
        self.sample = random.sample(self.dic[keys[0]][keys[1]][keys[2]][keys[3]], sample_size)

    def show_sample(self):
        self.clearLayout(self.gridLayout_2)
        labels=[]
        c=0

        for img_name in self.sample:
            i_label = QtWidgets.QLabel()
            # i_label.setText(img_name)

            sil_path = "../../datasets/casia_B1_silhouettes/{}-{}.png".format(self.key, img_name)
            img_path = "../../datasets/casia_B1_images/{}-{}.jpg".format(self.key, img_name)
            img = cv2.imread(img_path)
            sil = cv2.imread(sil_path)

            beta = (1.0-self.alpha)
            new_img = cv2.addWeighted(img, self.alpha, sil, beta, 0.0)
            #cv2 image converted to qt image
            qtimg = QtGui.QImage(new_img.data, img.shape[1], img.shape[0],QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap(QtGui.QPixmap.fromImage(qtimg))
            i_label.setPixmap(pixmap)
            labels.append(i_label)

            self.gridLayout_2.addWidget(i_label, c//3, c%3, 1, 1)
            c+=1
    
    def read_json(self,path):
        with open(path) as f:
            return json.load(f)

    def update_lists(self):
        self.com_lists.append(list(self.dic.keys()))
        self.com_lists.append(list(self.dic[self.curr_box[0]].keys()))
        
        for c_list in self.com_lists:
            c_list.sort()
        
    def update_combo(self):
        self.subBox.addItem("All")
        self.typeBox.addItem("All")
        self.subBox.addItems(self.com_lists[0])
        self.typeBox.addItems(self.com_lists[1])
        
        

    
    def update_curr_box(self):
        sub = []
        if(self.subBox.currentText() == "All"):
            sub = self.com_lists[0]
        else:
            sub.append(self.subBox.currentText())

        typ = []
        if(self.typeBox.currentText() == "All"):
            typ = self.com_lists[1]
        else:
            typ.append(self.typeBox.currentText())

        self.curr_box = [sub,typ]

        self.update_files()
        self.check_path = 'checklists/{}-{}.json'.format(self.subBox.currentText(),self.typeBox.currentText())
        self.read_list()
        # self.write_chechables()
        
        
    def update_files(self):
        self.files = []
        for sub in self.curr_box[0]:
            for typ in self.curr_box[1]:
                tns = list(self.dic[sub][typ].keys())
                tns.sort()
                for tn in tns:
                    angles = list(self.dic[sub][typ][tn].keys())
                    angles.sort()
                    for angle in angles:
                        self.files.append("{}-{}-{}-{}".format(sub,typ,tn,angle))
        
        self.list_files()
        self.totalFilesLabel.setText("N° of folders: {}".format(len(self.files)))
        



    def list_files(self):
        #Clear previous list
        self.model.removeRows( 0, self.model.rowCount() )
        #Create new list
        self.model.itemChanged.connect(self.on_check_changed)
        self.selModel = self.listView.selectionModel()
        self.selModel.currentChanged.connect(self.on_list_changed)
        for f in self.files:
            # Create an item with a caption
            item = QtGui.QStandardItem(f)
        
            # Add a checkbox to it
            
            item.setCheckable(True)
            item.setUserTristate(True)
        
            # Add the item to the model
            self.model.appendRow(item)
    
    def write_chechables(self):
        for i in range(self.model.rowCount()):
            key = self.model.item(i).text()
            value = self.is_useful[key]
            self.model.item(i).setCheckState(value)
    
    def clearLayout(self,layout):
        for i in reversed(range(layout.count())): 
            layout.itemAt(i).widget().setParent(None)

    def init_useful_dic(self):
        self.is_useful = {}
        for key in self.files:
            self.is_useful[key] = 0
    
    def save_list(self):
        with open(self.check_path, 'w') as f:
            json.dump(self.is_useful, f)
        msg = QtWidgets.QMessageBox()
        msg.setText("Lista guardada correctamente")
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.exec_()
    
    def read_list(self):
        print(self.check_path, os.path.exists(self.check_path))

        if os.path.exists(self.check_path):
            with open(self.check_path) as f:
                self.is_useful = json.load(f)
            self.write_chechables()
        else:
            self.init_useful_dic()
        


    
    def randomize(self):
        self.create_sample()
        self.show_sample()
    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
