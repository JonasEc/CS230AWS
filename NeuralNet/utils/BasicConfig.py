#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:54:27 2020

@author: jonasmg
"""

BATCH_SIZE = 32
IMGWIDTH = 128
IMGHEIGHT = 64
CHANNELS = 1
INPUTSHAPE =  [IMGHEIGHT,IMGWIDTH,1]
FULLCHARDICT = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'/':10,',':11, '+': 12, '-': 13, ' ': 14}
CHARDICT = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'/':10,',':11, '+': 12, '-': 13,}
INVDICT = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: '/', 11: ',', 12: '+', 13: '-'}
DICTLENGTH = len(CHARDICT) + 1 # for the empty string
MAXCHARLENGTH = 6

PARALLEL_INPUT_CALLS = 8

import sys, os
sys.path.append('/home/ubuntu/CS230AWS/NeuralNet/')
path  = '/home/ubuntu/CS230AWS/'

LOGDIR = os.path.join(os.curdir, "my_logs")
SAVEDIR = os.path.join(os.curdir, "my_models")

