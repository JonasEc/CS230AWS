#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 20:35:54 2020

@author: jonasmg
"""
 
import sys, os, pickle, time
sys.path.append('/home/jonasmg/Documents/OCRProject/NeuralNet/')
path  = '/home/jonasmg/Documents/OCRProject/'
import numpy as np

import tensorflow.compat.v2 as tf
from tensorflow import keras
from tensorflow.keras.layers import (Input, Flatten, Dense, Conv2D, MaxPool2D, 
                                GlobalAveragePooling2D, Activation, BatchNormalization,
                                Dropout)
from tensorflow.keras import backend as K

from utils.inputFinland import loadImgsWithLabels, DataGen
from utils.BasicConfig import (BATCH_SIZE, FULLCHARDICT,  INPUTSHAPE, LOGDIR, SAVEDIR)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

################################################################################################3
#%%
X, Y_label, Y_len,  N = loadImgsWithLabels(True)


Y_label = 1*(Y_label[:,0] != (FULLCHARDICT[' ']))*(Y_label[:,0]  != (FULLHARDICT['-']))
Y_label = Y_label.reshape((len(Y_label),1))


trainID = int(np.ceil(N*0.75))
X_train = X[:trainID]
Y_label_train = Y_label[:trainID]
X_dev = X[trainID:]
Y_label_dev = Y_label[trainID:]


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_label_train)).batch(BATCH_SIZE)
dev_dataset = tf.data.Dataset.from_tensor_slices((X_dev, Y_label_dev)).batch(BATCH_SIZE)

################################################################################################3
#%%

# Layer Params:   Filts K  Padding  Name     
layer_params = [ [  64, 7, 'same',  'conv1'], 
                 [  64, 3, 'same',  'conv2'], 
                 [ 128, 3, 'same',  'conv3'], 
                 [ 128, 3, 'same',  'conv4'], 
                 [ 256, 3, 'same',  'conv5'],
                 [ 256, 3, 'same',  'conv6'], 
                 [ 512, 3, 'same',  'conv7'], 
                 [ 512, 3, 'same',  'conv8'],
                 # [ 1024, 3, 'same',  'conv9'],
                 # [ 1024, 3, 'same',  'conv10'],
                 # [ 1024, 3, 'same',  'conv11'],
                 ] 
wherePool = [1,3,5]
# Dense Params:  neurs, drpot
DenseNets = [   [256, 0.5], 
                [128, 0.5], 
                # [64, 0.5],
            ]

# Network parameters
kernel_size = (3, 3)
pool_size = (2,2)
act = 'relu'
minibatch_size = BATCH_SIZE

inputs = Input(name='the_input', shape=INPUTSHAPE)
layer = layer_params[0]
inner = Conv2D(layer[0], layer[1], strides=(1,1), padding=layer[2], activation=act,
                     use_bias=False, kernel_initializer="he_normal", name=layer[3] )(inputs)    
inner = BatchNormalization(momentum=0.95, epsilon=1e-05, center=True, scale=True)(inner)
for i in range(1,len(layer_params)):
    layer = layer_params[i]
    inner = Conv2D(layer[0], layer[1], strides=(1,1), padding=layer[2], activation=act,
                     use_bias=False, kernel_initializer="he_normal", name=layer[3] )(inner)    
    inner = BatchNormalization(momentum=0.95, epsilon=1e-05, center=True, scale=True)(inner)
    
    if i in wherePool:  
       inner = MaxPool2D(pool_size=pool_size, strides=(2, 2), padding='same', name=f'pool{i+1}')(inner)

inner = GlobalAveragePooling2D(name='globalAvg')(inner)

for k in range(len(DenseNets)):
  inner = Dense(DenseNets[k][0], activation=act, name=f'dense{k}')(inner)
  inner = BatchNormalization(momentum=0.95, epsilon=1e-05, center=True, scale=True)(inner)
  inner = Dropout(DenseNets[k][1])(inner)

output = Dense(1, activation='sigmoid', name='OutputLayer')(inner)

sgd = keras.optimizers.SGD( learning_rate=0.02,
                            decay=1e-6,
                            momentum=0.9,
                            nesterov=True)

model = keras.Model(inputs=inputs, outputs=output)


model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])
model.summary()


################################################################################################3
#%%

run_name = 'BINARY'
if not os.path.exists(SAVEDIR):
            os.makedirs(SAVEDIR)
if not os.path.exists(LOGDIR):
            os.makedirs(LOGDIR)
if not os.path.exists(os.path.join(SAVEDIR, run_name)):
            os.makedirs(os.path.join(SAVEDIR, run_name))
if not os.path.exists(os.path.join(LOGDIR, run_name)):
            os.makedirs(os.path.join(LOGDIR, run_name))


early_stopping= keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)
checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(LOGDIR, run_name) + "/bestRunning_model.hdf5", 
                                             monitor='val_loss', verbose=1,
                                             save_best_only=True, mode='auto')
rlrop = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=3, verbose=1, mode='auto',
                                          min_delta=0.0001, cooldown=0, min_lr=0,)


# #%%
history = model.fit(train_dataset, epochs=80, 
                    validation_data = dev_dataset,
                    callbacks=[early_stopping, checkpoint,rlrop])

model.save(os.path.join(SAVEDIR, run_name) + "/last_model.h5")


