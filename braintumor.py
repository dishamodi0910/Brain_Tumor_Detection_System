
#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil
import math
import glob
import pickle
from keras.layers import Conv2D,MaxPool2D,Dropout,Flatten,Dense,BatchNormalization,GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.applications.mobilenet import preprocess_input
from keras.layers import Flatten,Dense
from keras.models import Model,load_model
from keras.applications.mobilenet import MobileNet
import streamlit as st
ROOT_DIR = "brain_tumor_dataset"
number_of_images = {};
#Finding the total number of images present in the both the types...
#Healthy and Unhealthy
#Yes -> Tumor is there
#No -> Tumor is absent
for dir in os.listdir(ROOT_DIR):
  number_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR,dir))) 

#number_of_images.items()
#Separating the test data, training data and validation data
#If the main folder doesn't exists, create it.
def DataModel(p,split):
  if not os.path.exists("./"+p):
    os.mkdir("./"+p)
    #Creating subfolders of the val, train and test data 
    for dir in os.listdir(ROOT_DIR):
      os.makedirs("./"+p+"/"+dir)
      
      #Adding the data randomly to the folders
      for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR,dir)),size = (math.floor(split*number_of_images[dir])-2),replace = False):
        O = os.path.join(ROOT_DIR,dir,img)
        D = os.path.join("./"+p,dir)
        shutil.copy(O,D)  #shutil is used to copy data from one folder to the another folder
        os.remove(O)
 
  else :
    print(f"{p} folder exists")

DataModel("train",0.7)      #Training data
DataModel("val",0.15)      #Validation data
DataModel("test",0.15)      #Testing data
#The main flow of model for processing images would be -->
#1. Creating the filters by removing the unwanted parts
#2. Pooling the filtered image as using it entirely is not possible
#3. Drop out the image a bit
#4. Flattening the image
st.write("Brain Tumor Detection System")
model = Sequential()
#Filtering the data
model.add(Conv2D(filters = 16,kernel_size = (3,3),activation = 'relu',input_shape = (224,224,3)))
#Filtering the data again
model.add(Conv2D(filters = 36,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPool2D(pool_size = (2,2)))     #MaxPool the data as taking it entirely is not possible
#Filtering the data again
model.add(Conv2D(filters = 64,kernel_size = (3,3),activation = 'relu'))
model.add(Conv2D(filters = 128,kernel_size = (3,3),activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#Drop out the unwanted parts
model.add(Dropout(rate = 0.25))
#Flattening the data to create connected layers
model.add(Flatten())
model.add(Dense(units=64,activation = 'relu'))
model.add(Dropout(rate = 0.25))
#Single fully connected layer
model.add(Dense(units=1,activation = 'sigmoid'))
model.summary()
#Compiling the sequential Model
model.compile(optimizer = 'adam',loss = keras.losses.binary_crossentropy,metrics = ['accuracy'])
def preprocessingImages(path):
   #input : path
   #output : pre processed image
 image_data = ImageDataGenerator(zoom_range = 0.2,shear_range = 0.2, preprocessing_function = preprocess_input ,horizontal_flip = True)
 image = image_data.flow_from_directory(directory=path,target_size = (224,224), batch_size = 32,class_mode = 'binary')

 return image

path = "train/"
train_data = preprocessingImages(path)

def preprocessingImagesNew(path):
    #input : path , output : pre processed image
  image_data = ImageDataGenerator(preprocessing_function= preprocess_input)
  image = image_data.flow_from_directory(directory=path,target_size = (224,224), batch_size = 32,class_mode = 'binary')

  return image

path = "test/"      #Testing phase
test_data = preprocessingImagesNew(path);
path = "val/"       #Validating data
val_data = preprocessingImagesNew(path)

#Early Stopping and Model check point...
from keras.callbacks import ModelCheckpoint, EarlyStopping
es = EarlyStopping(monitor = "val_accuracy",min_delta = 0.01,patience = 2,verbose = 1,mode = 'auto')
mc  = ModelCheckpoint(monitor = "val_accuracy",filepath = "./bestmodel.h5",verbose = 1,mode = 'auto')
cd = [es,mc]    #callbacks
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
his = model.fit_generator(generator=train_data,
                              steps_per_epoch = 1,
                              epochs = 10,
                              verbose = 1,
                              validation_data = val_data,
                              validation_steps = 16,
                              callbacks = cd)
h = his.history;
h.keys();
import matplotlib.pyplot as plt
plt.plot(h['val_accuracy'])
plt.plot(h['accuracy'])
accuracy = model.evaluate_generator(test_data)[1]
print(f"The accuracy of our model is : {accuracy*100}%")

pickle.dump(model,open('braintumor.pkl','wb'))
ld_model = pickle.load(open('braintumor.pkl','rb'))

from keras.models import load_model
from keras.applications.mobilenet import MobileNet, preprocess_input
model = load_model("bestmodel.h5")
from keras.utils.image_utils import img_to_array,load_img
from keras.applications.mobilenet import preprocess_input
path = st.file_uploader("Please upload an image ",type=["jpg","png"])
# img = load_img(path,target_size=(224,224))
def import_and_predict(path,model):
 input_arr = img_to_array(img)/255
 input_arr = preprocess_input(input_arr)
 i_arr = np.array([input_arr])
 i_arr.shape
 prediction = np.argmax(model.predict(i_arr))

 return prediction
#input_arr.shape
#input_arr = np.expand_dims(input_arr,axis = 0)
#prediction = model.predict(input_arr)[0][0]

if file is None:
  st.text("Please enter some image")
else:
 if(prediction[0]==0):
    st.write("Cancer detected");
 else:
    st.write("Cancer not detected");