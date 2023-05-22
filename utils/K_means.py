import matplotlib.pyplot as plt
import cv2
import numpy as np


class k_means:
     def __init__(self, cvImg):
         self.original_image = cvImg
         self.img = cv2.cvtColor(self.original_image,cv2.COLOR_BGR2RGB) 
         self.vectorized = self.img.reshape((-1,3)) 
         self.vectorized = np.float32(self.vectorized)
         
     def process(self, K):
         criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0) 
         attempts = 10 
         ret,label,center = cv2.kmeans(self.vectorized,K,None,criteria,attempts, cv2.KMEANS_RANDOM_CENTERS) 
         center = np.uint8(center) 
         res = center[label.flatten()] 
         self.result_image = res.reshape((self.img.shape))
         
         return self.result_image
         
         figure_size = 15 
         plt.figure(figsize=(figure_size,figure_size)) 
         plt.subplot(1,2,1),plt.imshow(self.img) 
         plt.title('Original Image'), plt.xticks([]), plt.yticks([]) 
         plt.subplot(1,2,2),plt.imshow(self.result_image) 
         plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([]) 
         plt.show()
         


class k_means:
    def __init__(self):
        
        print("test")
        
    def process(self, K):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0) 
        attempts = 10 
        ret,label,center = cv2.kmeans(self.vectorized,K,None,criteria,attempts, cv2.KMEANS_RANDOM_CENTERS) 
        center = np.uint8(center) 
        print("center",center)
        res = center[label.flatten()] 
        result_image = res.reshape((self.img.shape))
         
        return self.result_image
    
        