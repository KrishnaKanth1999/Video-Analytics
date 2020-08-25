from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
import shutil
import random
from keras.applications.vgg16 import VGG16
from keras.models import Sequential

root_dir = r'../gdrive/My Drive/New Blur Dataset/'
output_dir =r'../gdrive/My Drive/test_blur/'
ref = 1
files = [name for name in os.listdir(root_dir)]
for file in files:
  newpath = r'../gdrive/My Drive/test_blur/'+file
  if not os.path.exists(newpath):
      os.makedirs(newpath)
for name in files:
    print(root_dir+name)
    print('hi')
    for root, dirs, files in os.walk(root_dir+name):
        print(root)
        print(dirs)
        print(files)
        number_of_files = len(os.listdir(root))
        if number_of_files > ref:
            ref_copy = int(round(0.2 * number_of_files))
            for i in range(ref_copy):
                chosen_one = random.choice(os.listdir(root))
                file_in_track = root
                file_to_copy = file_in_track + '/' + chosen_one
                if os.path.isfile(file_to_copy) == True:
                    shutil.move(file_to_copy,output_dir+name)
                    print(file_to_copy)
        else:
            for i in range(len(files)):
                track_list = root
                file_in_track = files[i]
                file_to_copy = track_list + '/' + file_in_track
                if os.path.isfile(file_to_copy) == True:
                    shutil.move(file_to_copy,output_dir+name)
                    print(file_to_copy)
print('Finished !')







train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('../gdrive/My Drive/New Blur Dataset',
                                                 target_size=(224, 224),
                                                 batch_size=128,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('../gdrive/My Drive/test_blur',
                                            target_size=(224, 224),
                                            batch_size=128,
                                            class_mode='binary')



model = VGG16(weights='imagenet', include_top=True)
#
model.layers.pop()

for layer in model.layers:
     layer.trainable = False
modell = Sequential()
for layer in model.layers:
     modell.add(layer)
modell.add(Dense(1, activation='sigmoid'))

modell.compile(optimizer="adam", loss='binary_crossentropy',metrics=['accuracy'])
filepath = "blur1.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, mode='max', verbose=1)
from keras.callbacks import EarlyStopping

stop_here_please = EarlyStopping(patience=7)
callbacks_list = [checkpoint, stop_here_please]
modell.fit_generator(training_set,
                     steps_per_epoch=1330/ 128,
                      epochs=20,
                      validation_data=test_set,
                     validation_steps=1000 / 128, callbacks=callbacks_list)
#videotoframes

import glob,cv2
cap = cv2.VideoCapture("blurry cctv footage - YouTube.MP4")
success,cam = cap.read()
count =0
success=True
while success:
    cv2.imwrite('new blur/'+"Frame%d.jpg"%count,cam)
    #gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
  
   
    #cv2.imshow('Frame',cam)
    success,cam = cap.read()
    count+=1
  #cv2.imshow('frame',cam)

# =============================================================================
# for file in glob.glob("Frame/*.jpg"):
#     images = cv2.imread(file)
#  
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#       break
# cap.release()
# cv2.destroyAllWindows()
# =============================================================================
#duplicates
import subprocess
subprocess.call(["image-cleaner G:\\Honeywellhack\\Final Code\\noblur"])
import os
os.system("image-cleaner G:\\Honeywellhack\\Final Code")
import os
os.getcwd()
#Model Load

from keras.models import load_model  
from keras.layers import Dense
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
weights_model="../gdrive/My Drive/blur_new.h5"

model = VGG16(weights='imagenet', include_top=True)

model.layers.pop()

for layer in model.layers:
     layer.trainable = False
modell = Sequential()
for layer in model.layers:
     modell.add(layer)
modell.add(Dense(1, activation='sigmoid'))

modell.load_weights(weights_model)

#prediction

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

test=ImageDataGenerator().flow_from_directory(r"G:\Honeywellhack\Final Code\new blur",target_size = (224,224),shuffle=False)
# test_image = image.img_to_array(test)
# test_image = np.expand_dims(test_image, axis = 0)
y_pred=modell.predict_generator(test,steps=len(test)).flatten()
y_pred1=(y_pred>0.5)
if False in y_pred1:
        print("Blur")
else:
    print("No Blur")
for i in y_pred:
    print(i)
print(y_pred1[30])
print(len(y_pred))




