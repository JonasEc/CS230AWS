#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:10:28 2020

@author: jonasmg
"""

import cv2
import numpy as np

path = '/home/jonasmg/Documents/OCRProject/'
cens = path + 'ScannedCensus/1940/'
trans = path + 'ScannedCensus/Transcribed1940/'

quantThresh = 0.15
thresh = 0.75
shrinkR, shrinkC = 25, 25
grayThresh, grayMin = 125, 0.25

IMAGENO1NORMALIZERacross = 69
IMAGENO1NORMALIZERdown = 308
IMAGENO2NORMALIZERacross = 81
IMAGENO2NORMALIZERdown = 66

font = cv2.FONT_HERSHEY_SIMPLEX 
color = (255, 0, 0) # Blue color in BGR 



def rotate(img, angle):
  image_center = tuple(np.array(img.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)	
  return result

def normalizer(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	threshold = np.quantile(image[:,:,2].flatten(), quantThresh)
	mask =(image[:,:,2] >=threshold)
	image[mask] = [0,0,0]
	image[~mask] = [0,0,255]
	gray = image[:,:,2]
	lines = cv2.HoughLinesP(gray,5,np.pi/180, threshold=1000, minLineLength=750, maxLineGap=3) 
	lines = lines[:,0,:]

	k = lines[np.abs(lines[:, 0] - lines[:, 2]) < 10, : ]
	normalizingIndex = k[:,0].argmin()
	normalizingAddAcross = np.min([k[normalizingIndex,0],k[normalizingIndex,2]]) 	
	normalizingAngleAcross = np.arctan((k[normalizingIndex,0] - k[normalizingIndex,2])/np.abs((k[normalizingIndex,1] - k[normalizingIndex,3])))
	if k[normalizingIndex,1] < k[normalizingIndex,3]:
		 normalizingAngleAcross = -normalizingAngleAcross

	
	k = lines[np.abs(lines[:, 1] - lines[:, 3]) < 10, : ]
	l = np.abs(k[:, 0] - k[:, 2])
	k = k[l > 1000]
	normalizingIndex = k[:,1].argmin()
	normalizingAddDown = k[normalizingIndex,1]	
	normalizingAngleDown = np.arctan((k[normalizingIndex,1] - k[normalizingIndex,3])/np.abs((k[normalizingIndex,0] - k[normalizingIndex,2])))
	if k[normalizingIndex,1] < k[normalizingIndex,3]:
		normalizingAngleDown = -normalizingAngleDown

	return normalizingAddAcross, normalizingAddDown, normalizingAngleAcross, normalizingAngleDown

def readForm(imageNo):
	img = cv2.imread(cens+ f'{imageNo}.jpg', -1)
	imgsize = np.shape(img)

	black = 1*(img == [0,0,0]).any(axis=2)
	blackRows = black.sum(axis=1)
	blackCols = black.sum(axis=0)
	cropRows = 1 - 1*(blackRows > thresh*imgsize[1] )
	cropCols = 1 - 1*(blackCols > thresh*imgsize[0] )
	img = img[cropRows == 1, :, :]
	img = img[:, cropCols ==1, :]
	img = img[shrinkR:-shrinkR, shrinkC:-shrinkC, :]

	addAcross, addDown, angleAcross, angleDown = normalizer(img)
	if np.abs(angleAcross) >= 1:
		img = rotate(img, angleAcross)
# 	if np.abs(angleDown) >= 1:
# 		img = rotate(img, angleDown)		
	return img, addAcross, addDown 


	

#%%

def page1(img, addAcross, addDown,paint):
	dic = {}
	normAcross = addAcross- IMAGENO1NORMALIZERacross  
	normDown = addDown- IMAGENO1NORMALIZERdown
	
	START = ( 210 + normAcross, 500 + normDown)
	across = 163
	lastAcross= 214
	down = 87
	for s, row in enumerate(range(5,11)):
		for k, column in enumerate(['B','C','D','E','F','G','H']):
			tl = (START[0] + k*across, START[1] + s*down)
			if column == 'H':
				br = (START[0] + (k*across + lastAcross), START[1] + (s+1)*down)		
			else:
				br = (START[0] + (k+1)*across, START[1] + (s+1)*down)
			dic[(column +str(row))] =  (tl, br)
			if paint:
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
	 
		
	START = ( 1510 + normAcross, 582 + normDown) 
	across = 145
	down = 87
	for s, row in enumerate(range(13,18)):
		for k, column in enumerate(['B','C','D','E','F','G','H','I','J','K',"L","M","N"]):
			tl = (START[0] + k*across, START[1] + s*down)
			br = (START[0] + (k+1)*across, START[1] + (s+1)*down)
			dic[(column +str(row))] =  (tl, br)
			if paint:
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
	
	START = (165 + normAcross, 1175 + normDown) 
	acrossD = {'B':165,'C':165,'D':165,'E':162,'F':163,
			   'G':145, 'H':162,'I':162,'J':162,'K':145,"L":145,
			   "M":162,"N":162, "O":162, "P":162, "Q":162, "R":162,
			   "S":162, "T":162, "U":200}
	acrCum = 0
	down = 87
	for s, row in enumerate(range(20,25)):
		for k, column in enumerate(['B','C','D','E','F','G','H','I','J','K',"L",
									"M","N", "O", "P", "Q", "R", "S", "T", "U"]):
			if k < 10:
				offset = 0
			elif k < 15:
				offset = 1
			else:
				offset = 1
			tl = (START[0] + acrCum, START[1] + s*(down-offset))
			acrCum += acrossD[column]
			br = (START[0] + acrCum, START[1] - offset + (s+1)*(down-offset))
			dic[(column +str(row))] =  (tl, br)
			if paint:
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
		acrCum = 0
	
	START = (73 + normAcross, 1924 + normDown) 	
	acrossD = {'A': 157, 'B':157,'C':157,'D':156,'E':155,'F':160,
			   'G':160, 'H':160,'I':162,'J':160,'K':113,"L":113,
			   "M":113,"N":113, "O":113, "P":109, "Q":106, "R":110,
			   "S":110, "T":110, "U":110, "V": 208, "W": 110, "X": 208}
	acrCum = 0
	down = 87
	offset = 0
	row= 27
	for k, column in enumerate(['A','B','C','D','E','F','G','H','I','J','K',"L",
								"M","N", "O", "P", "Q", "R", "S", "T", "U",
								"V", "W", "X"]):
		if k > 15:
			offset = -4
		elif k > 17:
			offset = -6
		tl = (START[0] + acrCum, START[1] + offset)
		tlFont = (tl[0] + 15, tl[1] +30)
		acrCum += acrossD[column]
		br = (START[0] + acrCum, START[1] + down + offset)
		dic[(column +str(row))] =  (tl, br)
		if paint:
			img = cv2.rectangle(img, tl, br, color=color, thickness=2)
			img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
	
	
	START = (75 + normAcross, 2304 + normDown) 
	acrossD = {'A': 130, 'B':130,'C':130,'D':130,'E':130,'F':130,
			   'G':130, 'H':130,'I':110,'J':130,'K':130,"L":110,
			   "M":130,"N":130, "O":115, "P":125, "Q":110, "R":120,
			   "S":120, "T":120, "U":120, "V": 120, "W": 120, "X": 120,
			   "Y":120, "Z":120, "AA": 120}
	acrCum = 0
	down = 86
	offset = 0
	row = 30
	for k, column in enumerate(['A','B','C','D','E','F','G','H','I','J','K',"L",
								"M","N", "O", "P", "Q", "R", "S", "T", "U",
								"V", "W", "X", "Y", "Z", "AA"]):
		if k > 15:
			offset = -4
		elif k > 17:
			offset = -6
		tl = (START[0] + acrCum, START[1] + offset)
		acrCum += acrossD[column]
		br = (START[0] + acrCum, START[1] + down + offset)
		dic[(column +str(row))] =  (tl, br)
		if paint:
			tlFont = (tl[0] + 15, tl[1] +30)
			img = cv2.rectangle(img, tl, br, color=color, thickness=2)
			img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
	acrCum = 0
	
	return dic, img


def page2(img, addAcross, addDown, paint):
	dic ={}
	normAcross = addAcross- IMAGENO2NORMALIZERacross  
	normDown = addDown- IMAGENO2NORMALIZERdown
	
	START = (172 + normAcross, 322 + normDown) 
	acrossD = {'B':171,'C':171,'D':171,'E':171,'F':171,
			   'G':171, 'H':171,'I':168,'J':168,'K':168,"L":168,
			   "M":168,"N":168, "O":168, "P":168, "Q":165, "R":165,
			   "S":165, "T":165,}
	acrCum = 0
	down = 88
	for s, row in enumerate(range(2,7)):
		offset = 0
		for k, column in enumerate(['B','C','D','E','F','G','H','I','J','K',"L",
									"M","N", "O", "P", "Q", "R", "S", "T"]):
			if k >= 8:
				offset = 2
			elif k>= 12:
				offset = 4
			elif k>= 15:
				offset = 7				
			tl = (START[0] + acrCum, START[1] + s*down - offset)
			acrCum += acrossD[column]
			br = (START[0] + acrCum, START[1]+  (s+1)*down- offset )
			dic[(column +str(row))] =  (tl, br)
			if paint:
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
		acrCum = 0	

	START = (177 + normAcross, 999 + normDown) 
	acrossD = { 'B':113,'C':126,'D':126,'E':126,'F':135,
			   'G':129, 'H':130,'I':130,'J':130,'K':130,"L":130,
			   "M":130,"N":133, "O":129, "P":129, "Q":130, "R":130,
			   "S":130, "T":128, "U":129, "V": 128, "W": 127, "X": 130,
			   "Y":130, "Z":130,}
	acrCum = 0
	down = 87
	for s, row in enumerate(range(9,14)):
		offset = 0
		for k, column in enumerate(['B','C','D','E','F','G','H','I','J','K',"L",
									"M","N", "O", "P", "Q", "R", "S", "T", "U",
									"V", "W", "X", "Y", "Z"]):
			if k >= 8:
				offset = 2
			elif k>= 12:
				offset = 4
			elif k>= 15:
				offset = 7				
			tl = (START[0] + acrCum, START[1] + s*down - offset)
			acrCum += acrossD[column]
			br = (START[0] + acrCum, START[1]+  (s+1)*down- offset )
			dic[(column +str(row))] =  (tl, br)
			if paint:	
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
		acrCum = 0	
		
		
	START = (604 + normAcross, 1656 + normDown) 
	acrossD = { 'B':113,'C':113,'D':111,'E':109,'F':110, 'G':110, }
	downD = { 16:125,17:80,18:328,19:82}
	acrCum = 0
	downCum =0
	for s, row in enumerate(range(16,20)):
		first  = downCum
		downCum += downD[row]
		offset = 0
		for k, column in enumerate(['B','C','D','E','F','G']):			  
			tl = (START[0] + acrCum, START[1] + first)
			acrCum += acrossD[column]
			downCum 
			br = (START[0] + acrCum, START[1]+  downCum )
			dic[(column +str(row))] =  (tl, br)
			if paint:	
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
		acrCum = 0		   

	dic['A22'] = ((643 + normAcross, 2270 + normDown),(1244 + normAcross,2344 + normDown))
	dic['B22'] = ((643 + normAcross, 2344 + normDown),(1244 + normAcross,2465 + normDown))
	if paint:	
		img = cv2.rectangle(img, dic['A22'][0],  dic['A22'][1], color=color, thickness=2)
		img = cv2.putText(img, 'A22', ( (dic['A22'][0][0] +15),  (dic['A22'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['B22'][0],  dic['B22'][1], color=color, thickness=2)
		img = cv2.putText(img, 'B22', ( (dic['B22'][0][0] +15),  (dic['B22'][0][1] + 30)) , font,   1, color) 


	START = (1756 + normAcross,1610 + normDown)
	acrossD = { 'B':108,'C':110,'D':110,'E':110,'F':110, 'G':110, 'H':110}
	downD = { k:83 for k in range(25,35)}
	acrCum = 0
	downCum =0
	for s, row in enumerate(range(25,35)):
		first  = downCum
		downCum += downD[row]
		offset = 0
		for k, column in enumerate(['B','C','D','E','F','G', 'H']):			  
			tl = (START[0] + acrCum, START[1] + first)
			acrCum += acrossD[column]
			downCum 
			br = (START[0] + acrCum, START[1]+  downCum )
			dic[(column +str(row))] =  (tl, br)
			if paint:	
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
		acrCum = 0			   


	START = (2686 + normAcross,1766 + normDown)
	acrossD = { 'B':161,'C':163,'D':164,'E':215}
	downD = { k:87 for k in range(37,40)}
	acrCum = 0
	downCum =0
	for s, row in enumerate(range(37,40)):
		first  = downCum
		downCum += downD[row]
		offset = 0
		for k, column in enumerate(['B','C','D','E']):			  
			tl = (START[0] + acrCum, START[1] + first)
			acrCum += acrossD[column]
			downCum 
			br = (START[0] + acrCum, START[1]+  downCum )
			dic[(column +str(row))] =  (tl, br)
			if paint:	
				tlFont = (tl[0] + 15, tl[1] +30)
				img = cv2.rectangle(img, tl, br, color=color, thickness=2)
				img = cv2.putText(img, (column +str(row)), tlFont , font,   1, color) 
		acrCum = 0  

	row = 42
	dic['A42'] = ((2750 + normAcross, 1970 + normDown),(2950 + normAcross,2180 + normDown))
	dic['B42'] = ((2750 + normAcross, 2180 + normDown),(2950 + normAcross,2296 + normDown))
	dic['C42'] = ((2750 + normAcross, 2296 + normDown),(2950 + normAcross,2415 + normDown))
	dic['D42'] = ((3250 + normAcross, 2025 + normDown),(3375 + normAcross,2095 + normDown))
	dic['E42'] = ((3250 + normAcross, 2095 + normDown),(3375 + normAcross,2215 + normDown))
	dic['F42'] = ((3250 + normAcross, 2215 + normDown),(3375 + normAcross,2285 + normDown))
	dic['G42'] = ((3250 + normAcross, 2285 + normDown),(3375 + normAcross,2350 + normDown))
	dic['H42'] = ((3250 + normAcross, 2350 + normDown),(3375 + normAcross,2420 + normDown))
	if paint:
		img = cv2.rectangle(img, dic['A42'][0],  dic['A42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'A42', ( (dic['A42'][0][0] +15),  (dic['A42'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['B42'][0],  dic['B42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'B42', ( (dic['B42'][0][0] +15),  (dic['B42'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['C42'][0],  dic['C42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'C42', ( (dic['C42'][0][0] +15),  (dic['C42'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['D42'][0],  dic['D42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'D42', ( (dic['D42'][0][0] +15),  (dic['D42'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['D42'][0],  dic['E42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'E42', ( (dic['E42'][0][0] +15),  (dic['E42'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['E42'][0],  dic['F42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'F42', ( (dic['F42'][0][0] +15),  (dic['F42'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['F42'][0],  dic['G42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'G42', ( (dic['G42'][0][0] +15),  (dic['G42'][0][1] + 30)) , font,   1, color) 
		img = cv2.rectangle(img, dic['H42'][0],  dic['H42'][1], color=color, thickness=2)
		img = cv2.putText(img, 'H42', ( (dic['H42'][0][0] +15),  (dic['H42'][0][1] + 30)) , font,   1, color) 
	return dic, img	



#%%

def makeDicts(imageNo, paint=False):
	img, addAcc, addDown = readForm(imageNo)
	if int(imageNo)%2:
		dic, img = page1(img, addAcc,IMAGENO1NORMALIZERdown, paint)
	else:
		dic, img = page2(img, addAcc,IMAGENO2NORMALIZERdown, paint)
	if paint:
		cv2.imwrite(path + 'ScannedCensus/Marked/' + imageNo + '_marked.png',img)	
	return dic
	
dics = {}

for k in range(1,29):
	if k < 10:
		dics[k] = makeDicts('000' +str(k), True)
	else:
		dics[k] = makeDicts('00' +str(k), True)
		
		
import pickle

with open(path + 'ScannedCensus/dictionary1940.p', 'wb') as f:
	pickle.dump(dics, f)	
