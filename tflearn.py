from keras import models, layers
import os
import cv2
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d,max_pool_2d
from tflearn.layers.core import input_data,dropout,fully_connected
from tflearn.layers.estimator import regression

TRAIN_DIR = "C:/Users/Admin/Documents/NOTEBOOK/ASL/Images_Gray"
TRAIN_DIR = '/home/arjun/ARJUN/mv2/Dataset/cats_and_dogs/training_set/'

num_classes=2
IMG_SIZE = 200
LEARNING_RATE = 0.0001
MODEL_NAME = 'cat.model'

def vectorize_data(TRAIN_DIR):
    result = []
    labels = []
    for label in os.listdir(TRAIN_DIR):
        path=""
        path=os.path.join(TRAIN_DIR, label)
        for img in os.listdir(path):
            path2=""
            path2 = os.path.join(path, img)
            i = cv2.imread(path2)
            #i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
            
            i = cv2.resize(cv2.imread(path2, cv2.IMREAD_GRAYSCALE), (IMG_SIZE, IMG_SIZE))
            
            result.append(i)
            labels.append(label)
    
    return result, labels

x, y =vectorize_data(TRAIN_DIR)
x_train = np.array(x)
y_train = np.array(y)

x_train = np.expand_dims(x_train, axis=-1)
print(x_train.shape)

from keras.utils.np_utils import to_categorical

dictionary = {'cats':0, 'dogs':1}

keys, inv = np.unique(y_train, return_inverse=True)
print(keys,' ' ,inv)

vals = np.array([dictionary[key] for key in keys])
y_train_new = vals[inv]

y_train_new_cat = to_categorical(y_train_new, num_classes)

# SHUFFLE
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]
x_new,y_new = unison_shuffled_copies(x_train,y_train_new_cat)

tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
#Conv Layer 1
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 2
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 3
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 4
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 5
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
#Conv Layer 6
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.2)
#Fully Connected Layer with SoftMax as Activation Function
convnet = fully_connected(convnet, 2, activation='softmax')
#Regression for ConvNet with ADAM optimizer
convnet = regression(convnet, optimizer='adam', learning_rate=LEARNING_RATE, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

model.fit(x_train, y_new, n_epoch=10,  snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save('cat.model')
