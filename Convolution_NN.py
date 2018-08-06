# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 15:09:26 2018

@author: Aditya Vartak
"""

from keras.models import Sequential #to initialise NN as sequence of layers
from keras.layers import Conv2D#for convolution step
from keras.layers import MaxPooling2D#for max pooling
from keras.layers import Dense,Flatten

#Initialising  CNN
classifier=Sequential()
#add convolution layer(step 1)

classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))#no of filters/feature maps,rows,columns of feature maps,border_mode='same',input_shape is shape of input image

#add pooling layer(step 2)

classifier.add(MaxPooling2D(pool_size=(2,2)))
#we need convolution and max pooling step because not only it shortens the computation but also convey how the pixels are related to surrounding ones for a feature

#adding 2md convolution Layer

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#add Flattening layer(Step 3)
classifier.add(Flatten())#flattens the Matrix in 1d array to give it as input to ANN

#Step 4: Full Connection
classifier.add(Dense(units= 128, activation='relu'))
classifier.add(Dense(units=1 , activation='sigmoid'))#fully connected layer ,128 is trial and error value you can take any other value 

#compile the model

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

#since we dont have enough images for train to avoid overfitting we have to augment existing images to create random images
from keras.preprocessing.image import ImageDataGenerator
#from keras documentation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

"""
train_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_set,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=test_set,
        validation_steps=2000)

"""
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)
