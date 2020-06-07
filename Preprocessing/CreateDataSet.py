#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 17:08:15 2020

@author: jonasmg
"""

import numpy as np
import pandas as pd
import cv2 
import pickle

path = '/home/jonasmg/Documents/OCRProject/'
cens = path + 'ScannedCensus/1940/'
trans = path + 'ScannedCensus/Transcribed1940/'
colorCells = path + 'ScannedCensus/ExtractedC/'
preppedCells1 = path + 'ScannedCensus/ExtractedBW_V1/'
preppedCells2 = path + 'ScannedCensus/ExtractedBW_V2/'

dics = pickle.load(open(path + 'ScannedCensus/dictionary1940.p', 'rb'))
keys1 = [k for k in dics[1]]
keys2 = [k for k in dics[2]]

def pad(x):
	x = str(x)
	x = (4 - len(x))*'0' + x
	return x

replist = [pad(k) + '_' + k + '42' for k in ['D', 'E', 'F', 'G', 'H']]
def rep(x):
	if x == '1' or x == 1:
		return '+'
	elif x == '0' or x ==0:
		return '-'
	else:
		return x

#%%
	
l = []
for k in range(1,29):
	if k%2 == 0:
		df = pd.read_excel(trans  + pad(k) +'.xlsx', header=None)
		df = df.loc[1:, :25]
		df.columns = ['A', 'B','C','D','E','F',
				   'G', 'H','I','J','K',"L",
				   "M","N", "O", "P", "Q", "R",
				   "S", "T", "U", "V", "W", "X",
				   "Y", "Z"]
		df.index = df.index + 1
		df = df.loc[df['A'].notna(), :]
		df = df.loc[~df.index.isin([21,41]), :]
		df.fillna('*', inplace=True)
		df = df.stack().reset_index()
		df.columns = ['N', 'L', 'V']
		df['Cell'] = df['L'] + df['N'].astype('str')
		df.drop(['N','L'], axis=1, inplace=True)
		df =df.loc[df.Cell.isin(keys2)]
		df['Cell'] = pad(k) + '_' + df['Cell']
		df.loc[df['Cell'].isin(replist), 'V'] = df.loc[df['Cell'].isin(replist), 'V'].apply(rep) 
		
	else:
		df = pd.read_excel(trans  + pad(k) +'.xlsx', header=None)
		df = df.loc[4:, :26]
		df.columns = ['A', 'B','C','D','E','F',
				   'G', 'H','I','J','K',"L",
				   "M","N", "O", "P", "Q", "R",
				   "S", "T", "U", "V", "W", "X",
				   "Y", "Z", "AA"]
		df.index = df.index + 1
		df = df.loc[df['A'].notna(), :]
		df = df.loc[~df.index.isin([26,29]), :]
		df.fillna('*', inplace=True)
		df = df.stack().reset_index()
		df.columns = ['N', 'L', 'V']
		df['Cell'] = df['L'] + df['N'].astype('str')
		df.drop(['N','L'], axis=1, inplace=True)
		df =df.loc[df.Cell.isin(keys1)]
		df['Cell'] = pad(k) + '_' + df['Cell']
	
	df['V'] = df['V'].apply(lambda x: str(x).strip())
	l.append(df)
#%%	
DF = pd.concat(l)
DF.to_csv(path + 'ScannedCensus/FullData1940_RawwBrokenCells.csv', index=False)


DF['V'] = DF['V'].apply(lambda x: x.replace(' ', ''))
DF['V'] = DF['V'].apply(lambda x: x.replace('*', ''))
DF['V'] = DF['V'].apply(lambda x: x.replace('.', ','))
DF['V'] = DF['V'].apply(lambda x: x.replace('"', ''))

def check(x):
    for c in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
              'p','q','r','s','t','u','v','w','x','y','z', 'A',
              'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '?']:
        if c in x:
            return False
    return True

DF =DF.loc[DF.V.apply(lambda x: check(x)), :]
DF.loc[(DF.V.isna()) | (DF.V == ''), 'V'] = ' '
DF.to_csv(path + 'ScannedCensus/FullData1940.csv', index=False)
# 	df = df.dropna()
	
