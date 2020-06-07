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

CHARDICT = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'-':10,'/':11,',':12, '+': 13, ' ': 14}
INVDICT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '-', 11: '/', 12: ',', 13: '+', 14: ' '}
DICTLENGTH = len(CHARDICT) + 1 # for the empty string
IMGWIDTH = 128
IMGHEIGHT = 64
MAXCHARLENGTH = 6


def preproc(fn):
    img = cv2.imread(fn, 0)
    img = cv2.resize(img, (IMGWIDTH,IMGHEIGHT), interpolation = cv2.INTER_AREA)     
    img =  (255 - img)/255      
    return img.T


def makeExamples(labels, bw, N):
    start = input('From where?')
    start = int(start)
    labels['V'] = labels['V'].apply(lambda x: str(x))
    vals = labels['V'].tolist()
    cols = 2
    for k,f in enumerate(labels.Cell.tolist()[start:(start+N)]):
        X = preproc(bw + f +'.png')
        plt.subplot(N // cols, cols, k + 1)        
        plt.imshow(X.T, cmap='Greys_r')
        val = vals[start + k]
        plt.xlabel(f"Truth = {val} \n Cell = {f}" )
        fig = plt.gcf()
    fig.set_size_inches(10, 20)
    plt.savefig(path + f'ProcessedData/Examples/Test{start}.png' )
    plt.close()        

labels = pd.read_csv(path + 'ScannedCensus/FullData1940.csv')

makeExamples(labels, preppedCells2, 10)