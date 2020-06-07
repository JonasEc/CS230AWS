#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 22:38:46 2020

@author: jonasmg
"""


import cv2
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt

path  = '/home/ubuntu/CS230AWS/'

from utils.BasicConfig import (IMGWIDTH, IMGHEIGHT, CHARDICT, DICTLENGTH, MAXCHARLENGTH, PARALLEL_INPUT_CALLS)


def loadImgsWithLabels(full=False, synth=False):
	if full:
		name = 'ProcessedData/KerasReadyFull'
	else:
		name = 'ProcessedData/KerasReady'
	if synth:
		name = name + synth
	X	   = np.load(path + name + '0.npy', allow_pickle=True)	
	if synth:
		X = X.transpose([0,2,1,3])
	Y_label = np.load(path + name + '1.npy', allow_pickle=True)	
	Y_len   = np.load(path + name + '2.npy', allow_pickle=True)	
	N	   = np.load(path + name + '3.npy', allow_pickle=True)[0]	
	return X, Y_label,  Y_len,  N


class DataGen(keras.callbacks.Callback):
	def __init__(self, minibatch_size, SeqLength, X, Y_label, Y_len, N, currTrainIndex):
		self.minibatch_size = minibatch_size
		self.SeqLength = SeqLength
		self.X = tf.cast(X, tf.float32)
		self.N = N
		self.Y_label = Y_label
		self.Y_len = Y_len
		self.currTrainIndex = currTrainIndex
	
	@staticmethod			
	def image_erase_random(image):
		x = tf.random.uniform([1], minval=0, maxval=IMGWIDTH -5, dtype=tf.dtypes.int32)[0]
		y = tf.random.uniform([1], minval=0, maxval=IMGHEIGHT-5, dtype=tf.dtypes.int32)[0]
		patch = tf.zeros([4, 4])
		mask = tf.pad(patch, [[y, IMGHEIGHT-y -4], [x, IMGWIDTH -x -4]],
					  mode='CONSTANT', constant_values=1)
		image = tf.multiply(image, tf.reshape(mask, [IMGHEIGHT, IMGWIDTH, 1]))
		return image
		
	@staticmethod
	def image_add_random(image):
		x = tf.random.uniform([1], minval=0, maxval=IMGWIDTH -5, dtype=tf.dtypes.int32)[0]
		y = tf.random.uniform([1], minval=0, maxval=IMGHEIGHT-5, dtype=tf.dtypes.int32)[0]

		basepatch = tf.ones([2,4],dtype=tf.dtypes.float32)
		scalepath = tf.random.uniform([2,4],dtype=tf.dtypes.float32)
		patch = tf.multiply(basepatch,scalepath)
		mask = tf.pad(patch, [[y, IMGHEIGHT-y -2], [x, IMGWIDTH -x -4]] , 
					  mode='CONSTANT', constant_values=0)
		image = tf.add(image, tf.reshape(mask, [IMGHEIGHT, IMGWIDTH, 1]))
		image = tf.minimum(image, 1)
		return image

	@staticmethod
	def image_add_line(image):
		decider = tf.cast(tf.random.uniform([4], minval=0, maxval=2, dtype=tf.dtypes.int32) ,tf.dtypes.float32)
		
		ySa = tf.random.uniform(shape =[1], minval=0          , maxval=5          ,dtype=tf.dtypes.int32)[0]
		ySb = tf.random.uniform(shape =[1], minval=IMGHEIGHT-6, maxval=IMGHEIGHT-1,dtype=tf.dtypes.int32)[0]
		line1a = decider[0]*(tf.subtract(tf.ones([1,IMGWIDTH]),  tf.random.uniform([1,IMGWIDTH])/2))
		line1b = decider[1]*(tf.subtract(tf.ones([1,IMGWIDTH]),  tf.random.uniform([1,IMGWIDTH])/2))
		mask1a = tf.pad(line1a, [[ySa, IMGHEIGHT-ySa -1], [0, 0]],
					  mode='CONSTANT', constant_values=0)
		mask1b = tf.pad(line1b, [[ySb, IMGHEIGHT-ySb -1], [0, 0]],
					  mode='CONSTANT', constant_values=0)

		xSa = tf.random.uniform(shape =[1], minval=0,           maxval=5,          dtype=tf.dtypes.int32)[0]
		xSb = tf.random.uniform(shape =[1], minval=IMGWIDTH-6,  maxval=IMGWIDTH-1, dtype=tf.dtypes.int32)[0]			
		line2a = decider[2]*(tf.subtract(tf.ones([IMGHEIGHT,1]), tf.random.uniform([IMGHEIGHT,1])/2))
		line2b = decider[3]*(tf.subtract(tf.ones([IMGHEIGHT,1]), tf.random.uniform([IMGHEIGHT,1])/2))
		mask2a = tf.pad(line2a, [[0, 0], [xSa, IMGWIDTH-xSa -1]],
					  mode='CONSTANT', constant_values=0)
		mask2b = tf.pad(line2b, [[0, 0], [xSb, IMGWIDTH-xSb -1]],
					  mode='CONSTANT', constant_values=0)		

		image = tf.add(image, tf.reshape(mask1a, [IMGHEIGHT, IMGWIDTH, 1]))
		image = tf.add(image, tf.reshape(mask1b, [IMGHEIGHT, IMGWIDTH, 1]))
		image = tf.add(image, tf.reshape(mask2a, [IMGHEIGHT, IMGWIDTH, 1]))
		image = tf.add(image, tf.reshape(mask2b, [IMGHEIGHT, IMGWIDTH, 1]))
		image = tf.minimum(image, 1)		  
		return image
		
	@staticmethod
	def random_brightness(x, bmin, bmax):
		b = tf.add(tf.random.uniform(shape =[1], minval=bmin, maxval=bmax, dtype=tf.dtypes.float32)[0], 1)
		x = tf.multiply(x, b)
		x = tf.minimum(x, 1)
		return x


	@tf.function
	def image_bright_tf(self, tensor):
		return tf.map_fn(lambda x: self.random_brightness(x, -0.2, 0.2), tensor,
						 parallel_iterations=PARALLEL_INPUT_CALLS)	


	@staticmethod
	def image_rotate_random(image):
		def image_rotate_random_py_func(image, angle):
			# rand = np.random.randint(1000)
			# print(image.numpy().shape)
			# plt.imshow(image[:,:,0])
			# plt.savefig(f'/home/jonasmg/Pictures/in{rand}.png')
			rot_mat = cv2.getRotationMatrix2D((IMGWIDTH/2, IMGHEIGHT/2), angle, 1.0)
			rotated = cv2.warpAffine(image.numpy()[:,:,0], rot_mat, (IMGWIDTH,IMGHEIGHT))
			# print(rotated.shape)
			# plt.imshow(rotated)
			# plt.savefig(f'/home/jonasmg/Pictures/rot{rand}.png')
			r = rotated.reshape((IMGHEIGHT, IMGWIDTH, 1))
			# print(r.shape)
			# plt.imshow(r[:,:,0])
			# plt.savefig(f'/home/jonasmg/Pictures/rshaped{rand}.png')
			return r
		rand_amts = tf.maximum(tf.minimum(tf.random.normal([2], 0, .33), .9999), -.9999)
		angle = rand_amts[0] * 30  # degrees
		new_image = tf.py_function(image_rotate_random_py_func, (image, angle), tf.float32)
		return new_image

	@tf.function
	def image_rotate_random_tf(self, tensor):
		return tf.map_fn(self.image_rotate_random, tensor, parallel_iterations=PARALLEL_INPUT_CALLS)	

	@tf.function
	def image_erase_random_tf(self, tensor):
		return tf.map_fn(self.image_erase_random, tensor, parallel_iterations=PARALLEL_INPUT_CALLS)	

	@tf.function
	def image_add_random_tf(self, tensor):
		return tf.map_fn(self.image_add_random, tensor, parallel_iterations=PARALLEL_INPUT_CALLS)	

	@tf.function
	def image_add_line_tf(self, tensor):
		return tf.map_fn(self.image_add_line, tensor, parallel_iterations=PARALLEL_INPUT_CALLS)	 
	  
	def get_batch(self, r=False, training= True):
		while 1:
			if not r:
				curr = self.currTrainIndex
				new = (self.currTrainIndex +self.minibatch_size)
				if new >= self.N:
					curr = 0
					new = self.minibatch_size
					indices = tf.range(start=0, limit=tf.shape(self.X)[0], dtype=tf.int32)
					shuffled_indices = tf.random.shuffle(indices)
					self.X = tf.gather(self.X, shuffled_indices)
					self.Y_len = tf.gather(self.Y_len, shuffled_indices)
					self.Y_label = tf.gather(self.Y_label, shuffled_indices)
				X_use = self.X[curr:new, :, :, :]
				Y_label_use = self.Y_label[curr:new, :]
				Y_len_use = self.Y_len[curr:new]
				self.currTrainIndex = new
			else:
				random = np.random.randint(self.N - self.minibatch_size)
				X_use = self.X[random:(random+self.minibatch_size), :, :, :]
				Y_label_use = self.Y_label[random:(random+self.minibatch_size), :]
				Y_len_use = self.Y_len[random:(random+self.minibatch_size)]
			
			if training:
				for k in range(12):
					X_use = self.image_erase_random_tf(X_use)
					X_use = self.image_add_random_tf(X_use)
				X_use = self.image_add_line_tf(X_use)
				X_use = self.image_rotate_random_tf(X_use)
				X_use = self.image_bright_tf(X_use)
				
			inputs = {'the_input': X_use,
					  'the_labels': Y_label_use,
					  'input_length': np.ones([ self.minibatch_size,1])*self.SeqLength,
					  'label_length': Y_len_use,
					  }
			outputs = {'ctc': np.zeros([self.minibatch_size])} # dummy data for dummy loss function
			yield (inputs, outputs)	   



