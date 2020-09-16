import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

###########################load data#########################3
dir1 = "C:/Users/SamSung/Desktop/uni/y2/EDP/Angry"
dir2="C:Users/SamSung/Desktop/uni/y2/EDP/Hungry"
dir3="C:Users/SamSung/Desktop/uni/y2/EDP/What"
#생성하고 싶은 이미지들마다 다른 directory
labels = ["Angry", "Hungry", "What"]

train=[]
y=[]
tdata=[]


def cr_tdata():
    path = dir1
    class_num = labels.index(labels[0])#change for each folder
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(108,108))
            tdata.append([new_array, class_num])
        except:
            pass

cr_tdata()
random.shuffle(tdata)

for features, label in tdata:
    train.append(features)
    y.append(label)
train=np.asarray(train, dtype=np.int32)
trainz=train.reshape(-1,108*108)
train=np.divide(train,255)
#print(train[0].shape)

def sample_Z(batchsize):
    return np.random.normal(size=[batchsize,64])

def generator(Z,reuse=False):#Z is tensor of noise
    with tf.variable_scope('generator'):
        g1=tf.layers.dense(Z,batchsize,activation=tf.nn.relu)
        gout=tf.layers.dense(g1,batchsize)
    return gout

def discriminator(X,reuse=False):
    with tf.variable_scope('discriminator'):
        d1 = tf.layers.conv1d(tf.expand_dims(X,2),10,15,activation=tf.nn.leaky_relu,padding='same') #5 filters, #kernel size=5
		d2 = tf.layers.conv1d(d1,1,5,activation=tf.nn.leaky_relu,padding='same')
		dout = tf.layers.dense(tf.squeeze(d2,2),108*108) #returns 1 or 0
    return dout
