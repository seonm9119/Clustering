import sys
import cv2
from utils.FCM_cImg import FCM_cImg
import matplotlib.pyplot as plt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from utils.functions import k_means_cImg
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap



form_class = uic.loadUiType("main_window.ui")[0]


class PopUpWindow(QDialog):
    def __init__(self,parent):
        super(PopUpWindow, self).__init__(parent)
        Popup_ui = 'FCM_widget.ui'
        uic.loadUi(Popup_ui, self)


        self.fig = plt.Figure() 
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.verticalLayout.addWidget(self.canvas)
        self.verticalLayout.addWidget(self.toolbar)
        
        self.data_plot(parent)

        
        
    def data_plot(self,parent):
        
        ax = self.fig.gca(projection='3d')
        ax.scatter(parent.df["X"], parent.df["Y"], parent.df["R"], alpha=.8)  
        ax.scatter(parent.df["X"], parent.df["Y"], parent.df["G"], alpha=.8) 
        ax.scatter(parent.df["X"], parent.df["Y"], parent.df["B"], alpha=.8)           
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('RGB', fontsize=15)
        ax.legend(fontsize=15)
        plt.savefig('test.png')
        plt.show()

        
        self.canvas.draw() 
        self.show()
    


    


class MyWindow(QMainWindow, form_class):
    def __init__(self):
        
        self.cvImg=None
        self.height=None
        self.width=None
        self.channel = None
        
                
        super().__init__()
        self.setupUi(self)
        self.initUI()
        
        
    def initUI(self):
       
       #File
       self.actionOpen.triggered.connect(self.showDialog)
       
       #Data
       self.actionPlot.triggered.connect(self.data_plot)
       
       #Clustering
       self.actionOriginal.triggered.connect(self.original_clustering_cImg)
       #self.actionAutoencoder.triggered.connect(self.autoencoder_clustering)
       
              
       #self.runFCM.clicked.connect(self.FCM)
       #self.runEnFCM.clicked.connect(self.EnFCM)
     
       
    def showDialog(self):
       fname = QFileDialog.getOpenFileName(self, 'Open file', './')

       if fname[0]:
            
           self.original_image = cv2.imread(fname[0])
           self.grayImg = cv2.imread(fname[0],0)
           self.row, self.col, self.channel = self.original_image.shape
           self.n_data = self.row * self.col
           
            
           qPixmap = QPixmap()
           qPixmap.load(fname[0])
           self.pictureBox1.setPixmap(qPixmap)

        
     
        
    def original_clustering_cImg(self):
        
 
        
        n_cluster= int(self.spinBoxFCM_cluster.value())
        max_iter = int(self.spinBoxFCM_max_iter.value())
        m = float(self.doubleSpinBoxFCM_m.value())
        
        
        k_means_result = k_means_cImg(self.original_image, n_cluster, max_iter)
        fcm = FCM_cImg(self.original_image, n_cluster, max_iter, m)
        fcm_result = fcm.run()

        cv2.imwrite('Result/k-means.jpg', k_means_result)
        self.display('Result/k-means.jpg')



    def data_plot(self):

        self.df = prepare_data(self.row,self.col, self.cvImg)


        PopUpWindow(self)




        
    def btn_clicked(self):
        
        n_cluster= int(self.option.spinBox.value())
        m = float(self.option.doubleSpinBox.value())
        print(n_cluster, m)
        self.option.close()
        
   
        
    def display (self, img):
        
        
        
        qPixmap = QPixmap()
        qPixmap.load(img)
        self.pictureBox2.setPixmap(qPixmap)
      
    

      
                

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MyWindow()
    myWindow.show()
    app.exec_()