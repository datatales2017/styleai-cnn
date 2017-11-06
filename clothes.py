# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:11:40 2017

@author: manasa
"""
import keras
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten

model=Sequential()
model.add(Conv2D(64,(3,3),input_shape=(64, 64, 3),activation="relu")) #Convolutional Layer added
model.add(MaxPooling2D(pool_size=(2,2)))#Maxpooling step

model.add(Conv2D(64,(3,3),input_shape=(64, 64, 3),activation="relu")) #Convolutional Layer added
model.add(MaxPooling2D(pool_size=(2,2)))#Maxpooling step

model.add(Flatten())#Flatten step-this is the initial layer

#full connection
model.add(Dense(units=128,activation="relu"))#First Hidden Layer
model.add(Dense(units=4,activation="softmax"))#Output Layer

#Compiling
model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=['accuracy'])

#Image Preprocessing-Image Augmentation
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
        'S:/Gallery/Train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')

test = test_datagen.flow_from_directory(
        'S:/Gallery/Test',
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical')

model.fit_generator(train,
                    steps_per_epoch=2280, #number of images in training set
                    epochs=5,
                    validation_data=test,
                    nb_val_samples=400)
                    
#Saving the model
from sklearn.externals import joblib

joblib.dump("model", "Clothesnn.pkl", compress=0, protocol=None, cache_size=None)
loaded_model = joblib.load("Clothesnn.pkl")
