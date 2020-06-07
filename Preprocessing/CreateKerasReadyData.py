#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:08:15 2020

@author: jonasmg
"""

import numpy as np
import pandas as pd
import cv2 

import matplotlib.pyplot as plt

path = '/home/jonasmg/Documents/OCRProject/'
preppedCells1 = path + 'ScannedCensus/ExtractedBW_V1/'
preppedCells2 = path + 'ScannedCensus/ExtractedBW_V2/'

FULLCHARDICT = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'/':10,',':11, '+': 12, '-': 13, ' ': 14}
CHARDICT = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'/':10,',':11, '+': 12, '-': 13,}
INVDICT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '/', 11: ',', 12: '+', 13: '-'}
DICTLENGTH = len(CHARDICT) + 1 # for the empty string
IMGWIDTH = 128
IMGHEIGHT = 64
MAXCHARLENGTH = 6

def preproc(fn):
    img = cv2.imread(fn, 0)
    img = cv2.resize(img, (IMGWIDTH,IMGHEIGHT), interpolation = cv2.INTER_AREA)     
    img =  (255 - img)/255      
    return img

def labler(x, allUsed=False):
    ret = []
    for c in x:
        if allUsed:
            ret.append(FULLCHARDICT[str(c)])
        else:
            ret.append(CHARDICT[str(c)])
    return ret

def MergeImgsWithLabels(labels, bw, allUsed=False):
    if not allUsed:
        labels = labels.loc[~labels['V'].isin(['-', ' ', '']),:]
    labels['V'] = labels['V'].apply(lambda x: str(x))
    N = len(labels)
   
    X = np.zeros([N, IMGHEIGHT, IMGWIDTH])
    Y_len = np.zeros([N,1])
    Y_label = np.ones([N , MAXCHARLENGTH]) * -1
    for k,f in enumerate(labels.Cell.tolist()):
        X[k, :, :] = preproc(bw + f +'.png')
        Y_len[k] = len(labels.iloc[k, 0])
        Y_label[k, 0:int(Y_len[k])] = labler(labels.iloc[k, 0], allUsed)
    X = X.reshape([N, IMGHEIGHT, IMGWIDTH, 1])
    # randomOrder = np.random.permutation(np.array(range(N)))
    # X = X[randomOrder]
    # Y_len = Y_len[randomOrder]
    # Y_label = Y_label[randomOrder]    
    return X, Y_label,  Y_len,  N

import pickle

labels = pd.read_csv(path + 'ScannedCensus/FullData1940.csv')
X, Y_label,  Y_len,  N =  MergeImgsWithLabels(labels, preppedCells2)
for i,m in enumerate([X, Y_label,  Y_len,  np.array([N])]):
    np.save(path + f'ProcessedData/KerasReady{i}.npy', m)
randomOrder = np.random.permutation(len(X))
pickle.dump(randomOrder, open(path + f'ProcessedData/orderRestr.p', 'wb'))

X, Y_label,  Y_len,  N =  MergeImgsWithLabels(labels, preppedCells2, True)
for i,m in enumerate([X, Y_label,  Y_len,  np.array([N])]):
    np.save(path + f'ProcessedData/KerasReadyFull{i}.npy', m)
randomOrder = np.random.permutation(len(X))
pickle.dump(randomOrder, open(path + f'ProcessedData/orderFull.p', 'wb'))



