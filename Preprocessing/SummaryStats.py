#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:51:53 2020

@author: jonasmg
"""
import matplotlib.pyplot as plt

import pandas as pd

DF = pd.read_csv('/home/jonasmg/Documents/OCRProject/ScannedCensus/FullData1940_RawwBrokenCells.csv')

print(len(DF))
DF['V'] = DF['V'].apply(lambda x: x.strip())
DF['V'] = DF['V'].apply(lambda x: x.replace(' ', ''))
DF['V'] = DF['V'].apply(lambda x: x.replace('\n', ''))
DF['V'] = DF['V'].apply(lambda x: x.replace('\t', ''))
DF['V'] = DF['V'].apply(lambda x: x.replace('*', ''))
DF['V'] = DF['V'].apply(lambda x: x.replace('.', ','))
DF['V'] = DF['V'].apply(lambda x: x.replace('"', ''))
DF['V'] = DF['V'].apply(lambda x: x.strip())

def check(x):
    for c in ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o',
              'p','q','r','s','t','u','v','w','x','y','z', 'A',
              'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
              'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '?']:
        if c in x:
            return True
    return False

print(len(DF.loc[DF.V.apply(lambda x: check(x)), :]))

DF.loc[(DF.V.isna()) | (DF.V == ' '), 'V'] = ''
DF['V'] = DF['V'].apply(lambda x: str(x))

print(len(DF[DF.V == '']))

print(len(DF[DF.V == '-']))

DF['Len'] = DF['V'].apply(lambda x: len(x))


DF = DF[~DF.V.apply(check)]
DF = DF[~(DF.V == '-')]

#%%
fig = plt.figure()
plt.hist(DF.loc[~DF['V'].isin(['', '-']),'Len'], bins=[1,2,3,4,5,6,7])	  
fig.savefig('/home/jonasmg/Documents/OCRProject/WriteUpMaterials/LengthDist_NonEmpty.png')

#%%

DFvals = DF.V.tolist()

DFvals = ''.join(DFvals)



from collections import Counter
count = Counter(DFvals)


setChars = list(set(DFvals))
setChars.sort()

v = []
h = []
for c in setChars:
	v.append(c)
	h.append(count[c])
plt.bar(v,h)

	  