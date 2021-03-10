#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter

import matplotlib.pyplot as plt
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze attention.''')
parser.add_argument('--attention_dir', nargs=1, type= str, default=sys.stdin, help = '''path to attention.''')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test/bench on.')




#FUNCTIONS
def analyze_type_attention(activations, bench_pred_types, bench_true_types):
    '''Analyze the activations per type
    {'NO_SP':0,'SP':1,'TAT':2,'LIPO':3}
    SP: Sec substrates cleaved by SPase I (Sec/SPI),
    LIPO: Sec substrates cleaved by SPase II (Sec/SPII),
    TAT: Tat substrates cleaved by SPase I (Tat/SPI).
    '''

    types = {'NO_SP':0,'Sec/SPI':1,'Tat/SPI':2,'Sec/SPII':3}

    all_type_activations = []
    for type in types:
        type_P = np.argwhere(bench_true_types==types[type])
        type_pred_P = np.argwhere(bench_pred_types==types[type])
        type_TP = np.intersect1d(type_P,type_pred_P)
        #type_FP =
        type_activations = activations[type_TP]
        av_type_activation = np.average(type_activations,axis=0)
        all_type_activations.append(av_type_activation)

    all_type_activations = np.array(all_type_activations)
    fig,ax = plt.subplots(figsize=(18/2.54,9/2.54))
    plt.imshow(all_type_activations)
    plt.yticks(ticks=[0,1,2,3],labels=[*types.keys()])
    plt.colorbar(orientation='horizontal')
    plt.tight_layout()
    plt.show()
    pdb.set_trace()

######################MAIN######################
args = parser.parse_args()
attention_dir = args.attention_dir[0]
checkpointdir=args.checkpointdir[0]
test_partition = args.test_partition[0]

#Parse
activations1 = []
activations2 = []
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #Load
    activations1.append(np.load(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations1.npy',allow_pickle=True))
    activations2.append(np.load(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations2.npy',allow_pickle=True))
#Array conversion
activations1 = np.array(activations1)
activations2 = np.array(activations2)
#Average across validation splits
activations1 = np.average(activations1,axis=0)
activations2 = np.average(activations2,axis=0)
#Average across embedding dimensions
activations1 = np.average(activations1,axis=2)
activations2 = np.average(activations2,axis=2)
#join the activations from both heads
activations = (activations1+activations2)/2
#Get predicted types and true types
bench_pred_types = np.load(checkpointdir+'bench_pred_types.npy',allow_pickle=True)
bench_true_types = np.load(checkpointdir+'bench_true_types.npy',allow_pickle=True)
#Analyze the activations for different types
analyze_type_attention(activations, bench_pred_types, bench_true_types)
pdb.set_trace()
