# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 19:35:45 2019

@author: Krishna Kanth
"""

import cv2,os
from keras.preprocessing import image
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
threshold = [100,200,400,500,700,900,1200,1300,1400,1500,1600,1700,1800,1900,2000]



def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

folderpathBlur = "G:/Honeywellhack/Blur/New Blur Dataset//Blur/"
folderpathNoBlur = "G:/Honeywellhack/Blur/New Blur Dataset//NoBlur/"

# load image arrays
y=[]
t1 = []

len_blur=len(os.listdir(folderpathBlur))
len_noblur=len(os.listdir(folderpathNoBlur))
for i in range(len_blur):
    y.append(1)
for i in range(len_noblur):
    y.append(0)
print(len(y))

for filename in  os.listdir(folderpathBlur):
        


        imagepath = folderpathBlur  + filename
        img = image.load_img(imagepath,target_size=(775,775))

        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        t1.append(fm)

        
for filename in os.listdir(folderpathNoBlur):
        


        imagepath = folderpathNoBlur + filename
        img = image.load_img(imagepath, target_size=(775,775))

        gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        t1.append(fm)
print("1")
for i in threshold:
    y_pred=[]
    for fm in t1:
            if fm < i:
                y_pred.append(1)
            else:
                y_pred.append(0)
  

    print(i,accuracy_score(y,y_pred))
    print(confusion_matrix(y,y_pred))
    print("--------------------")

    
            
            
    