#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 11:11:56 2020

@author: jonasmg
"""


import cv2
import numpy as np

path = '/home/jonasmg/Documents/OCRProject/'
cens = path + 'ScannedCensus/1940/'
trans = path + 'ScannedCensus/Transcribed1940/'
colorCells = path + 'ScannedCensus/ExtractedC/'
preppedCells1 = path + 'ScannedCensus/ExtractedBW_V1/'
preppedCells2 = path + 'ScannedCensus/ExtractedBW_V2/'

import pickle

dics = pickle.load(open(path + 'ScannedCensus/dictionary1940.p', 'rb'))


def pad(x):
	x = str(x)
	x = (4 - len(x))*'0' + x
	return x

#%%

# set pixels to zero based on quantiles:
hueThresh = 8
allCellsQuantile = 0.20
pageLevelQuantile = 0.15
cellLevelQuantile = 0.25

# eliminate black border around full picture based on:
blackThresh = 0.75
shrinkR, shrinkC = 25, 25


bufferX = 0
bufferY = 0
cutOff = 5
quantacross,quantdown = 0.98, 0.98 
threshAcross, threshDown = 0.85, 0.85
# mult = 0.55


#%%
def readAndReduce(imageNo):
	img = cv2.imread(cens+ f'{imageNo}.jpg', -1)
	imgsize = np.shape(img)

	black = 1*(img == [0,0,0]).any(axis=2)
	blackRows = black.sum(axis=1)
	blackCols = black.sum(axis=0)
	cropRows = 1 - 1*(blackRows > blackThresh*imgsize[1] )
	cropCols = 1 - 1*(blackCols > blackThresh*imgsize[0] )
	img = img[cropRows == 1, :, :]
	img = img[:, cropCols ==1, :]
	img = img[shrinkR:-shrinkR, shrinkC:-shrinkC, :]
	
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	pageThreshVal = np.quantile(imgHSV[:,:,2].flatten(), pageLevelQuantile)
	return img, pageThreshVal

def extractCells(imageNo, write=True):
	img, pageThreshVal = readAndReduce(imageNo)
	dic = dics[int(imageNo)]
	running = []
	cells = {}
	for k in dic:
		top, bottom = dic[k]
		x0, y0 = top
		x1, y1 = bottom
		snapC = img[(y0-bufferY):(y1+bufferY), (x0-bufferX):(x1+bufferX)]
		if write:
			cv2.imwrite(colorCells + f'{imageNo}_{k}.png',snapC)	
		cells[k] = snapC
		imgHSV = cv2.cvtColor(snapC, cv2.COLOR_BGR2HSV)
		running.append(imgHSV[:,:,2].flatten())
	allCells = np.concatenate( running, axis=0 )
	cellThreshVal = np.quantile(allCells, allCellsQuantile)
	return cellThreshVal, pageThreshVal, cells



def deleteLines(img, relative):
	shape = img.shape
	mask = (255 - img)/255
	valsAcross =[]
	for l in range(shape[0]):
		curVal = sum(mask[l,:] >0)/shape[1]
		valsAcross.append(curVal)
	valsDown =[]
	for l in range(shape[1]):
		curVal = sum(mask[:,l] >0)/shape[0]
		valsDown.append(curVal)

	if relative:
		thresholdAcross = np.quantile(valsAcross, quantacross) 
		thresholdDown = np.quantile(valsDown, quantdown) 
	else:
		thresholdAcross = threshAcross
		thresholdDown = threshDown

# 	for l in range(2,shape[0]):
# 		if valsAcross[l] >= thresholdAcross:
# 			img[l,:] = img[l-1, :]
# 		elif valsAcross[l-1] >= mult*thresholdAcross and valsAcross[l] >= mult*thresholdAcross:
# 			img[l-1,:] = img[l-2, :]
# 			img[l,:] = img[l-1, :]
	for l in range(1,shape[0]):
		if valsAcross[l] >= thresholdAcross:
 			img[l,:] = img[l-1, :]
	for l in range(2,shape[0]):
		if valsAcross[shape[0]-l] >= thresholdAcross:
 			img[shape[0]-l,:] = img[shape[0]-l+1, :]
	 
	if valsAcross[0] >= thresholdAcross:
		img[0,:] = 255
# 	if valsAcross[1] >= thresholdAcross:
# 		img[1,:] = 0   
	    
# 	for l in range(2,shape[1]):
# 		if valsDown[l] >= thresholdDown:
# 			img[:,l] = img[:, l-1]
# 		elif valsDown[l-1] >= mult*thresholdDown and valsDown[l] >= mult*thresholdDown:
# 			img[:,l-1] = img[:,l-2]
# 			img[:,l] = img[:,l-1]

	for l in range(1,shape[1]):
		if valsDown[l] >= thresholdDown:
 			img[:,l] = img[:, l-1]
	for l in range(2,shape[1]):
		if valsDown[shape[1]-l] >= thresholdDown:
 			img[:,shape[1]-l] = img[:, shape[1]-l+1]	 

	if valsDown[0] >= thresholdDown:
		img[:,0] = 255
# 	if valsDown[1] >= thresholdDown:
# 		img[:,1] = 0
		
	return img


def processImage(img, red=True, thresh=False, lines=True, relative=False, cut=False):
	image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	if thresh:
		threshold = thresh
	else:
		threshold = np.quantile(image[:,:,2].flatten(), cellLevelQuantile)
	if red:
		mask =((image[:,:,2] >=threshold) | (image[:,:,0] <= hueThresh))
	else:
		mask =(image[:,:,2] >=threshold)
		
	image[mask] = [0,0,255]
	gray = image[:,:,2]	

	if cut:
		imgShp = gray.shape
		gray = gray[cutOff:(imgShp[0]-cutOff), cutOff:(imgShp[1]-cutOff)]
	
	if lines:
		gray = deleteLines(gray, relative)
	grayMin = np.min(gray)
	g = np.rint(255*((gray - grayMin)/(255 - grayMin)))	
	
	return g


    

def loopCells(imageNo, red=True, thresh=False, relative=False, write=True):
	imageNo = pad(imageNo)
	img, pageThreshVal = readAndReduce(imageNo)
	cellThreshVal, pageThreshVal, cells = extractCells(imageNo, write)
	dicThresh = {'cell': False, 'globalCell': cellThreshVal, 'page': pageThreshVal}
	for k in dics[int(imageNo)]:
		cell = cells[k]
# 		cell = cv2.imread(colorCells + f'{imageNo}_{k}.png', -1)
		gray = processImage(cell, red, dicThresh[thresh], True, relative, False)
		cv2.imwrite(preppedCells1 + f'{imageNo}_{k}.png',gray)	
		gray = processImage(cell, red, dicThresh[thresh], False, False, False)
		cv2.imwrite(preppedCells2 + f'{imageNo}_{k}.png',gray)	
		
	
# def createPageThresh
# 		cv2.imwrite(preppedCells1 + f'{imageNo}_{k}.png',snapBW)	
		

#%%
for imageNo in range(1,29):
	loopCells(imageNo,True,'globalCell', False, True)

