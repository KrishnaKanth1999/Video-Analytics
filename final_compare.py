# import the necessary packages
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import glob
import operator
# =============================================================================
# %matplotlib inline
# =============================================================================

ssi=[]
fram=[]
dif=[]
fi=[]
count=0
imageB = cv2.imread("Frame0.jpg")
import os
# load the two input images
for file in os.listdir("newblur/New folder"):
    imageA = cv2.imread(file)
    # convert the images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    print(imageA)

    # compute the Structural Similarity Index (SSIM) between the two
    # images, ensuring that the difference image is returned
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    ssi.append(score)
    fram.append(imageA)
    dif.append(diff)
    fi.append(file)
    if(ssi[count]==1.0):
      print("SSIM: {}".format(score))
      fil=fi[count].split("/")
      l=len(fil)
      fil=fil[l-1].split(".")
      print("The input image is part of the video at %s"%fil[0])
    count+=1
     
    # compute the Structural Similarity Index (SSIM) between the two
print(len(ssi))
if 1.0 in ssi:
    ssi.remove(1.0)
index, value = max(enumerate(ssi), key=operator.itemgetter(1)) 

thresh = cv2.threshold(dif[index], 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
imageA=cv2.imread(fi[index])
he,we,_=imageA.shape
print(he,we)
print("The below images are %d percent similar"%(ssi[index]*100))
# loop over the contours
for c in cnts:
  # compute the bounding box of the contour and then draw the
  # bounding box on both input images to represent where the two
  # images differ
  (x, y, w, h) = cv2.boundingRect(c)
  #print("area=%f"%h*w)
  if (he*we)/(h*w)<=1900:
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
# show the output images
cv2.imshow("imageA",imageA)
#plt.show()
cv2.imshow("imageB",imageB)
#plt.show()
#cv2.imshow("Difference",diff)
#plt.show()
#plt.imshow(thresh)
#plt.show()
#cv2.imwrite('../gdrive/My Drive/output/'+"out1.jpg",imageA)
#cv2.imwrite('../gdrive/My Drive/output/'+"out2.jpg",imageB)
cv2.waitKey(0)
