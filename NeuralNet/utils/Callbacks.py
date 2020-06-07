#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:52:59 2020

@author: jonasmg
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
import os

from utils.BasicConfig import (CHARDICT, DICTLENGTH, 
                               MAXCHARLENGTH, LOGDIR, SAVEDIR, INVDICT)

from tensorflow import keras
from tensorflow.keras import backend as K

def labels_to_text(labels):
    ret = []
    for c in labels:
        if c == len(INVDICT):  # CTC Blank
            ret.append("")
        elif c == -1:
            ret.append('')
        else:
            ret.append(INVDICT[c])
    return "".join(ret)

def decode_batch(test_func, word_batch):
    out = test_func([word_batch])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret

class VizCallback(keras.callbacks.Callback):
    def __init__(self, run_name, test_func, valgenerator):
        self.test_func = test_func
        self.output_dir = os.path.join(LOGDIR, run_name)
        self.valgenerator = valgenerator
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def on_epoch_end(self, epoch, logs={}):
        # self.model.save_weights(
        #     os.path.join(self.output_dir, 'weights%02d.h5' % (epoch)))
        # self.show_edit_distance(256)
        word_batch = next(self.valgenerator)[0]
        res = decode_batch(self.test_func, word_batch['the_input'])
        cols = 2
        for i in range(8):
            plt.subplot(8 // cols, cols, i + 1)
            the_input = word_batch['the_input'][i, :, :, 0].numpy()
            plt.imshow(the_input, cmap='Greys_r')
            truth = labels_to_text(word_batch['the_labels'][i].numpy())
            decoded = res[i]
            plt.xlabel(f"Truth = {truth} \nDecoded = {decoded}" )
        fig = plt.gcf()
        fig.set_size_inches(10, 13)
        plt.savefig(os.path.join(self.output_dir, 'e%02d.png' % (epoch)))
        plt.close()
