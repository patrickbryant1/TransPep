#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze attention.''')
parser.add_argument('--attention_dir', nargs=1, type= str, default=sys.stdin, help = '''path to attention.''')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test/bench on.')




#FUNCTIONS
def analyze_type_attention(activations, bench_pred_types, bench_true_types, bench_seqs, test_partition, attention_dir):
    '''Analyze the activations per type
    {'NO_SP':0,'SP':1,'TAT':2,'LIPO':3}
    SP: Sec substrates cleaved by SPase I (Sec/SPI),
    LIPO: Sec substrates cleaved by SPase II (Sec/SPII),
    TAT: Tat substrates cleaved by SPase I (Tat/SPI).
    '''

    types = {'No SP':0,'Sec/SPI':1,'Tat/SPI':2,'Sec/SPII':3}
    amino_acids = np.arange(21)

    all_type_activations = []
    all_type_seqs = []
    for type in types:
        type_P = np.argwhere(bench_true_types==types[type])
        type_pred_P = np.argwhere(bench_pred_types==types[type])
        type_TP = np.intersect1d(type_P,type_pred_P)
        #type_FP =
        type_activations = activations[type_TP]
        type_seqs = bench_seqs[type_TP]

        #Go through all positions
        for i in range(70):
            col = type_seqs[:,i]
            #Go through all amino acids
            for aa in np.unique(col):
                col_pos = np.argwhere(col==aa)
                aa_activations = type_activations[col_pos,i]
                pdb.set_trace()



    #Array conversion
    #all_type_activations = np.array(all_type_activations)
    #all_type_seqs = np.array(all_type_seqs)
    pdb.set_trace()
    fig,ax = plt.subplots(figsize=(18/2.54,9/2.54))
    im = plt.imshow(all_type_activations)
    plt.yticks(ticks=[0,1,2,3],labels=[*types.keys()])
    plt.xlabel('Sequence position')
    plt.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(attention_dir+'type_attention_'+str(test_partition)+'.png',format='png',dpi=300)
    pdb.set_trace()

######################MAIN######################
plt.rcParams.update({'font.size': 7})
args = parser.parse_args()
attention_dir = args.attention_dir[0]
checkpointdir=args.checkpointdir[0]
test_partition = args.test_partition[0]

#Parse
enc_attention = []
enc_dec_attention = []
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #Load
    enc_attention.append(np.load(attention_dir+'enc_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))
    enc_dec_attention.append(np.load(attention_dir+'enc_dec_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))
#Array conversion
enc_attention = np.array(enc_attention)
enc_dec_attention = np.array(enc_dec_attention)
pdb.set_trace()
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
#Get seqs
bench_seqs = np.load(checkpointdir+'bench_seqs.npy',allow_pickle=True)
#Analyze the activations for different types
analyze_type_attention(activations, bench_pred_types, bench_true_types,bench_seqs,test_partition, attention_dir)
pdb.set_trace()
