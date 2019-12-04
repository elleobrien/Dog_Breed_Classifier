#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:24:34 2019
This script uses transfer learning to classify dogs as belonging to one of two breeds.
Hidden convolutional layers of the VGG-16 pre-trained model are used to generate features for each image, 
and then fully-connected layers are trained on top to label data.
@author: eobrien
"""


from keras.applications import VGG16
from keras import models, layers, optimizers
from keras.applications.vgg16 import preprocess_input
import numpy as np
import os 
import re
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from numpy.random import seed
from tensorflow import set_random_seed
import random as rn


################## FOR REPRODUCIBILITY #############################
# Set all random number seeds
seed(1)
set_random_seed(1)
rn.seed(1)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


############ Set some image processing parameters ####################
size = 224 # Default input size for VGG16

# Load in the convolutional base
conv_base = VGG16(weights='imagenet', 
                  include_top=False,
                  input_shape=(size, size, 3))  



# What are the two classes? They're coded in file names so use that here.
file_names = os.listdir("data")
classes = list(set([re.split(r"/|_|.jpg",name)[0] for name in file_names]))
class_dict = {classes[0]:0, classes[1]:1}

# How many files are there in the model directory?
n_files = len(file_names)

def extract_features(sample_count):
    features = np.zeros(shape=(sample_count, 7, 7, 512))  # Must be equal to the output of the convolutional base
    labels = np.zeros(shape=(sample_count))
    
    # Pass data through convolutional base
    
    # Stream each image from data folder and transform 
    for i in range(0,sample_count):
        file_loc = os.path.join("data",file_names[i])
        img = image.load_img(file_loc, target_size=(size,size))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)
        x = preprocess_input(x)
        
        
        # Predict VGG features for this image
        features[i] = conv_base.predict(x)
    # Get the class label
        label_code = re.split(r"/|_|.jpg",file_loc)[1]
        labels[i] = class_dict[label_code]
        
        if i % 10 == 0 and i > 0:
            print("Now processing image " + str(i) + " of " + str(sample_count) + ".")

    return features, labels
    
features, labels = extract_features(n_files)  
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2,
                                                    stratify = labels, random_state = 41)


# Define fully connected feed-forward model and its hyperparameters
epochs = 4
model = models.Sequential()
model.add(layers.Flatten(input_shape=(7,7,512))) 
model.add(layers.Dense(256,activation='relu', input_dim=(7*7*512)))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# Compile model
model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['acc'])

history = model.fit(X_train, Y_train,
                    epochs=epochs,
                    batch_size=32, 
                    validation_data=(X_test, Y_test))
					
final_val_acc = history.history['val_acc'][-1]

print("Validation Accuracy: %1.3f" % (final_val_acc))

model.save("model.h5")
