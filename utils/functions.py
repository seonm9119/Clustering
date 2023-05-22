import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from tensorflow import keras



def prepare_img(original_image):
    img = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB) 
    row, col, channel = img.shape    
    x = img.reshape((-1,3))
    x = np.float32(x)    
    return x, img


def axis_data(row, col):
    x_axis = np.arange(row)
    y_axis = np.arange(col)
    y_m,x_m =np.meshgrid(y_axis,x_axis)

    x_axis = x_m.reshape(row*col,1)
    y_axis = y_m.reshape(row*col,1)

    xy = np.column_stack([x_axis, y_axis])
    return xy


def plot_img(img, result_image, n_cluster, name):
    figure_size = 15 
    fig=plt.figure(figsize=(figure_size,figure_size)) 
    plt.subplot(1,2,1)
    plt.imshow(img) 
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])     
    plt.subplot(1,2,2),plt.imshow(result_image) 
    plt.title( name +' Segmented Image when c = %i' %n_cluster), plt.xticks([]), plt.yticks([]) 
    plt.savefig('Result/'+ name +'.png')
    plt.show()
    
    

def k_means_cImg(original_image, n_cluster, max_iter):
     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0) 
     max_iter = 10 
     x, img =prepare_img(original_image)
     
     ret,label,center = cv2.kmeans(x,n_cluster,None,criteria, max_iter, cv2.KMEANS_RANDOM_CENTERS) 
     center = np.uint8(center) 
     res = center[label.flatten()] 
     result_image = res.reshape((img.shape))
     
     #plot_img(img, result_image, n_cluster, 'K-means')
         
     return result_image
    


def autoencoder(dims, n_cluster, act='relu', init='glorot_uniform'):
    
    encoder = keras.models.Sequential([keras.layers.Input(shape=[dims]),                                           
                                               keras.layers.Dense(500, activation=act),
                                               keras.layers.Dense(500, activation=act),
                                               keras.layers.Dense(2000, activation=act),
                                               keras.layers.Dense(n_cluster, activation=act)])


    decoder = keras.models.Sequential([keras.layers.Dense(2000, activation=act, input_shape=[n_cluster]),
                                               keras.layers.Dense(500, activation=act),
                                               keras.layers.Dense(500, activation=act),
                                               keras.layers.Dense(dims, activation=act)])
    
    ae = keras.models.Sequential([encoder, decoder])
    return ae, encoder, decoder




def prepare_data(row, col, img):
      
    data = {}
    df = pd.DataFrame(data)

                
    for x in range(row):
        for y in range(col):
            interm = {'X':x,
                      'Y':y,
                      'R':img[x,y,0],
                      'G':img[x,y,1],
                      'B':img[x,y,2]}

            df = df.append(interm, ignore_index=True)
    
    return df

    

    

    
    
    