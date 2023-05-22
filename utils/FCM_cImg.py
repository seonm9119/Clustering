import numpy as np
import random
import cv2
from utils.functions import prepare_img, plot_img

SMALL_VALUE = 0.0001

class FCM_cImg:


    def __init__(self, original_image, n_cluster, max_iter, m):
        
        
        #------------------Check inputs----------------------------------
        if n_cluster <= 0 or n_cluster != int(n_cluster):
            raise Exception("<n_clusters> needs to be positive integer.")
        if m < 1:
            raise Exception("<m> needs to be >= 1.")
        #----------------------------------------------------------------
    
        self.original_image = original_image
        self.row, self.col, self.channel = original_image.shape
        self.n_cluster = n_cluster
        self.m = m   
        self.max_iter = max_iter
        self.n_data = self.row * self.col
        self.center = np.zeros((self.n_cluster, 1))
        
        self.x, self.img = prepare_img(original_image)
        self.x = self.x.T

        

        
    def init_membership_random(self): 

   
        self.u = np.zeros((self.n_data, self.n_cluster)) #uik (cxN)
        for k in range(self.n_data):
            row_sum = 0.0
            tmp = random.sample(range(0,self.n_cluster),self.n_cluster)
            for i in range(self.n_cluster):
                if i == self.n_cluster-1:  # last iteration
                    self.u[k][tmp[i]] = 1.0 - row_sum
                else:
                    rand_num = random.random() #0~1 난수
                    rand_num = round(rand_num, 2) #난수 반올림
                    if rand_num + row_sum <= 1.0:  # to prevent membership sum for a point to be larger than 1.0
                        self.u[k][tmp[i]] = rand_num
                        row_sum += self.u[k][tmp[i]]
        


    
    def compute_cluster_centers(self):
               

        interm1 = np.dot(self.x,self.u**self.m)
        interm2 = np.sum(self.u**self.m,axis=0)
        interm2[np.where(interm2 <= 0)]=SMALL_VALUE
 
        self.center = (interm1/interm2)



    def update_membership(self):

        dis =0.0
        for i in range(3):
            center,x = np.meshgrid(self.center[i],self.x[i])
            dis += (x-center)**2
        
            
        dis[np.where(dis <= 0)]=SMALL_VALUE             
        power = 1./(self.m-1)
        interm1 = dis**power
        interm2 = np.sum((1./dis)**power,axis=1)
        
        
        interm1[np.where(interm1 <= 0)]=SMALL_VALUE
        interm2[np.where(interm2 <= 0)]=SMALL_VALUE
    
        self.u = 1./(interm1*interm2[:,None])
        
        
        
    
    def run(self):
        
        self.init_membership_random()
        
        for i in range(self.max_iter):
            
            old_u = self.u.copy()
            self.compute_cluster_centers()
            self.update_membership()
            error = np.sum(abs(self.u-old_u))
            error = error.max()
            
            if error < 1e-5: break;
            
        
        label = []
        for k in range(self.n_data):
            label.append(self.u[k].argmax())
        
        label = np.array(label)
        print(label.shape)
        center = np.uint8(self.center.T)
        #center = center[:,2:5]
        print("FCMcenter",center)
        print(center.shape)
        res = center[label]
        
        result_image = res.reshape((self.img.shape))
        #plot_img(self.img, result_image, self.n_cluster, 'FCM')
        
        return result_image
        #return 0

             
    
