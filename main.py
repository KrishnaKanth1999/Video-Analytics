# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 07:04:08 2019

@author: Krishna Kanth
"""
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model  
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import glob,cv2

from sklearn.cluster import KMeans
import timeit
import webcolors
import sys
sys.path.append('/libsvm-3.18/python')
import numpy as np
import math as m
import sys
# for gamma function, called 
from scipy.special import gamma as tgamma
import os

from svm import *
from svmutil import *
import svm
import svmutil

# =============================================================================
# from final_image_quality import quality_score 
# =============================================================================

import numpy as np

weights_model="G:\Honeywellhack\Final Code/blur_new.h5"
model = VGG16(weights='imagenet', include_top=True)
model.layers.pop()
for layer in model.layers:
     layer.trainable = False
modell = Sequential()
for layer in model.layers:
     modell.add(layer)
modell.add(Dense(1, activation='sigmoid'))
modell.load_weights(weights_model)

blur_tracker=[]
color_names=[]
quality_tracker=[]

def AGGDfit(structdis):
    # variables to count positive pixels / negative pixels and their squared sum
    poscount = 0
    negcount = 0
    possqsum = 0
    negsqsum = 0
    abssum   = 0

    poscount = len(structdis[structdis > 0]) # number of positive pixels
    negcount = len(structdis[structdis < 0]) # number of negative pixels
    
    # calculate squared sum of positive pixels and negative pixels
    possqsum = np.sum(np.power(structdis[structdis > 0], 2))
    negsqsum = np.sum(np.power(structdis[structdis < 0], 2))
    
    # absolute squared sum
    abssum = np.sum(structdis[structdis > 0]) + np.sum(-1 * structdis[structdis < 0])

    # calculate left sigma variance and right sigma variance
    lsigma_best = np.sqrt((negsqsum/negcount))
    rsigma_best = np.sqrt((possqsum/poscount))

    gammahat = lsigma_best/rsigma_best
    
    # total number of pixels - totalcount
    totalcount = structdis.shape[1] * structdis.shape[0]

    rhat = m.pow(abssum/totalcount, 2)/((negsqsum + possqsum)/totalcount)
    rhatnorm = rhat * (m.pow(gammahat, 3) + 1) * (gammahat + 1)/(m.pow(m.pow(gammahat, 2) + 1, 2))
    
    prevgamma = 0
    prevdiff  = 1e10
    sampling  = 0.001
    gam = 0.2

    # vectorized function call for best fitting parameters
    vectfunc = np.vectorize(func, otypes = [np.float], cache = False)
    
    # calculate best fit params
    gamma_best = vectfunc(gam, prevgamma, prevdiff, sampling, rhatnorm)

    return [lsigma_best, rsigma_best, gamma_best] 

def func(gam, prevgamma, prevdiff, sampling, rhatnorm):
    while(gam < 10):
        r_gam = tgamma(2/gam) * tgamma(2/gam) / (tgamma(1/gam) * tgamma(3/gam))
        diff = abs(r_gam - rhatnorm)
        if(diff > prevdiff): break
        prevdiff = diff
        prevgamma = gam
        gam += sampling
    gamma_best = prevgamma
    return gamma_best

def compute_features(img):
    scalenum = 2
    feat = []
    # make a copy of the image 
    im_original = img.copy()

    # scale the images twice 
    for itr_scale in range(scalenum):
        im = im_original.copy()
        # normalize the image
        im = im / 255.0

        # calculating MSCN coefficients
        mu = cv2.GaussianBlur(im, (7, 7), 1.166)
        mu_sq = mu * mu
        sigma = cv2.GaussianBlur(im*im, (7, 7), 1.166)
        sigma = (sigma - mu_sq)**0.5
        
        # structdis is the MSCN image
        structdis = im - mu
        structdis /= (sigma + 1.0/255)
        
        # calculate best fitted parameters from MSCN image
        best_fit_params = AGGDfit(structdis)
        # unwrap the best fit parameters 
        lsigma_best = best_fit_params[0]
        rsigma_best = best_fit_params[1]
        gamma_best  = best_fit_params[2]
        
        # append the best fit parameters for MSCN image
        feat.append(gamma_best)
        feat.append((lsigma_best*lsigma_best + rsigma_best*rsigma_best)/2)

        # shifting indices for creating pair-wise products
        shifts = [[0,1], [1,0], [1,1], [-1,1]] # H V D1 D2

        for itr_shift in range(1, len(shifts) + 1):
            OrigArr = structdis
            reqshift = shifts[itr_shift-1] # shifting index

            # create transformation matrix for warpAffine function
            M = np.float32([[1, 0, reqshift[1]], [0, 1, reqshift[0]]])
            ShiftArr = cv2.warpAffine(OrigArr, M, (structdis.shape[1], structdis.shape[0]))
            
            Shifted_new_structdis = ShiftArr
            Shifted_new_structdis = Shifted_new_structdis * structdis
            # shifted_new_structdis is the pairwise product 
            # best fit the pairwise product 
            best_fit_params = AGGDfit(Shifted_new_structdis)
            lsigma_best = best_fit_params[0]
            rsigma_best = best_fit_params[1]
            gamma_best  = best_fit_params[2]

            constant = m.pow(tgamma(1/gamma_best), 0.5)/m.pow(tgamma(3/gamma_best), 0.5)
            meanparam = (rsigma_best - lsigma_best) * (tgamma(2/gamma_best)/tgamma(1/gamma_best)) * constant

            # append the best fit calculated parameters            
            feat.append(gamma_best) # gamma best
            feat.append(meanparam) # mean shape
            feat.append(m.pow(lsigma_best, 2)) # left variance square
            feat.append(m.pow(rsigma_best, 2)) # right variance square
        
        # resize the image on next iteration
        im_original = cv2.resize(im_original, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return feat

# function to calculate BRISQUE quality score 
# takes input of the image path
def test_measure_BRISQUE(dis):
    # read image from given path
# =============================================================================
#     dis = cv2.imread(imgPath, 1)
# =============================================================================

    # convert to gray scale
    dis = cv2.cvtColor(dis, cv2.COLOR_BGR2GRAY)

    # compute feature vectors of the image
    features = compute_features(dis)

    # rescale the brisqueFeatures vector from -1 to 1
    x = [0]
    
    # pre loaded lists from C++ Module to rescale brisquefeatures vector to [-1, 1]
    min_= [0.336999 ,0.019667 ,0.230000 ,-0.125959 ,0.000167 ,0.000616 ,0.231000 ,-0.125873 ,0.000165 ,0.000600 ,0.241000 ,-0.128814 ,0.000179 ,0.000386 ,0.243000 ,-0.133080 ,0.000182 ,0.000421 ,0.436998 ,0.016929 ,0.247000 ,-0.200231 ,0.000104 ,0.000834 ,0.257000 ,-0.200017 ,0.000112 ,0.000876 ,0.257000 ,-0.155072 ,0.000112 ,0.000356 ,0.258000 ,-0.154374 ,0.000117 ,0.000351]
    
    max_= [9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672, 1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368, 0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133, 3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270, 1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209, 0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484]

    # append the rescaled vector to x 
    for i in range(0, 36):
        min = min_[i]
        max = max_[i] 
        x.append(-1 + (2.0/(max - min) * (features[i] - min)))
    
    # load model 
    model = svmutil.svm_load_model("allmodel")

    # create svm node array from python list
    x, idx = gen_svm_nodearray(x[1:], isKernel=(model.param.kernel_type == PRECOMPUTED))
    x[36].index = -1 # set last index to -1 to indicate the end.
	
	# get important parameters from model
    svm_type = model.get_svm_type()
    is_prob_model = model.is_probability_model()
    nr_class = model.get_nr_class()
    
    if svm_type in (ONE_CLASS, EPSILON_SVR, NU_SVC):
        # here svm_type is EPSILON_SVR as it's regression problem
        nr_classifier = 1
    dec_values = (c_double * nr_classifier)()
    
    # calculate the quality score of the image using the model and svm_node_array
    qualityscore = svmutil.libsvm.svm_predict_probability(model, x, dec_values)

    return qualityscore

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name  

def video_capture(path):
    cap = cv2.VideoCapture(path)
    success,cam = cap.read()
    counter =0
    success=True
    cap = cv2.VideoCapture(path)
    length = int(cap.get(cv2.CAP_PROP_FPS))
    print( "Total FPS",length )
    while success:
        

        if counter==0:
            print("The resolution is ",cam.shape[:2])
            
        if counter%50==0:
        #cv2.imshow('Frame',cam)
            detect_color(cam,counter)
            print("-----------------------------------------------------")
            print("THe colors present are",color_names)
                 
       
            blur_prediction(cam)
            if 0 in blur_tracker:
                    print("There is a Blur in the video")
            else:
                    print("There is no Blur in the video")
        
            qualityscore = test_measure_BRISQUE(cam)
            quality_tracker.append(qualityscore)
            print("Score of the given image: ", qualityscore)
            
            success,cam = cap.read()
            print("-----------------------------------------------------")
        counter+=1
        
        
        
            
       
    
   
# =============================================================================
#         print("Score of the given image: ", qualityscore)
# =============================================================================
    if len(quality_tracker): 
        print("Overall Score of the given image: ",sum(quality_tracker)/len(quality_tracker))
        

def blur_prediction(direct):
    print("Blur Prediciton")
    #prediction
# =============================================================================
#     test_image = image.load_img(direct, target_size = (224, 224))
# =============================================================================
    test_image = cv2.resize(direct, (224,224)) 
# =============================================================================
#     test_image = image.img_to_array(resized_image)
# =============================================================================
    test_image = np.expand_dims(test_image, axis = 0)
    result=modell.predict_classes(test_image).flatten()
    blur_tracker.append(result)
    
    return result
        
def detect_color(direct,counter):
    
    #convert to rgb from bgr
    img = cv2.cvtColor(direct, cv2.COLOR_BGR2RGB)
            
    #reshaping to a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))

    
    #using k-means to cluster pixels
    if counter==0:
            a=12
    else:
          a=2
    kmeans = KMeans(n_clusters = a)
    kmeans.fit(img)
    
    for  c in kmeans.cluster_centers_:
        actual_name, closest_name = get_colour_name(tuple(c))
        if closest_name not in color_names:
                color_names.append(closest_name)
                
       
        
  
    
video_capture(r"G:\Honeywellhack\Hackathon_VideoSet/Frame skip.mp4")


    
    

   