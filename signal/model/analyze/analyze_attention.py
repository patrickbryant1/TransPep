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




#####################FUNCTIONS#####################
def get_pred_types(pred_annotations):
    '''Get the predicted types based on the annotations
    '''

    annotation_type_conversion = {0:1,1:2,2:3} #S(0)=SP(1), T(1)=TAT(2),L(2)=LIPO(3) - all other 0 (No SP)
    pred_types = []
    for i in range(len(pred_annotations)):
        if (0 in pred_annotations[i]) or (1 in pred_annotations[i]) or (2 in pred_annotations[i]):
            counts = Counter(pred_annotations[i])
            keys = [*counts.keys()]

            key_count=0 #Count the occurance of each annotation - take the max for the type
            key_type = 0
            for key in annotation_type_conversion: #Got through all keys
                if key not in keys:
                    continue
                else:
                    if counts[key]>key_count:
                        key_count=counts[key]
                        key_type=annotation_type_conversion[key]

            #Save
            pred_types.append(key_type)

        else:
            pred_types.append(0)

    return np.array(pred_types)


def analyze_type_attention(enc_dec_attention, seqs, true_types, pred_types,pred_annotations, attention_dir):
    '''Analyze the activations per type
    {'NO_SP':0,'SP':1,'TAT':2,'LIPO':3}
    SP: Sec substrates cleaved by SPase I (Sec/SPI),
    LIPO: Sec substrates cleaved by SPase II (Sec/SPII),
    TAT: Tat substrates cleaved by SPase I (Tat/SPI).
    '''

    types = {'NO_SP':0,'Sec/SPI':1,'Tat/SPI':2,'Sec/SPII':3}
    amino_acids = np.arange(21)

    for type in types:
        type_P = np.argwhere(true_types==types[type])
        type_pred_P = np.argwhere(pred_types==types[type])
        type_TP = np.intersect1d(type_P,type_pred_P)

        type_enc_dec_attention = enc_dec_attention[type_TP]
        type_seqs = seqs[type_TP]

        fig,ax = plt.subplots(figsize=(9/2.54,9/2.54))
        im = plt.imshow(np.average(type_enc_dec_attention,axis=0))#In seqs on x, out annotations on y
        plt.xlabel('Sequence position')
        plt.ylabel('Annotation position')
        plt.title(type)
        plt.tight_layout()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(attention_dir+'enc_dec_attention_'+str(types[type])+'.png',format='png',dpi=300)
        continue
        #Go through all positions
        for i in range(70):
            col = type_seqs[:,i]
            #Go through all amino acids
            for aa in np.unique(col):
                col_pos = np.argwhere(col==aa)
                aa_activations = type_activations[col_pos,i]
                pdb.set_trace()


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
test_partition = args.test_partition[0]

#Parse
enc_attention = []
enc_dec_attention = []
pred_annotations = []
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #Load
    try:
        enc_attention.append(np.load(attention_dir+'enc_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))
        enc_dec_attention.append(np.load(attention_dir+'enc_dec_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))
        pred_annotations.append(np.load(attention_dir+'pred_annotations_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))
    except:
        continue
#Array conversion
enc_attention = np.array(enc_attention)
enc_dec_attention = np.array(enc_dec_attention)
pred_annotations = np.array(pred_annotations)
#Average across validation splits
enc_attention = np.average(enc_attention,axis=0)
enc_dec_attention = np.average(enc_dec_attention,axis=0)
pred_annotations = np.average(pred_annotations,axis=0)
pred_annotations = np.argmax(pred_annotations,axis=2)
#Get types
pred_types = get_pred_types(pred_annotations)
#Max across attention heads
enc_attention = np.max(enc_attention,axis=1)
enc_dec_attention = np.max(enc_dec_attention,axis=1)

#Get true annotations and types
true_annotations = np.load(attention_dir+'annotations_'+str(test_partition)+'.npy',allow_pickle=True)
true_types = np.load(attention_dir+'types_'+str(test_partition)+'.npy',allow_pickle=True)
#Get seqs
seqs = np.load(attention_dir+'seqs_'+str(test_partition)+'.npy',allow_pickle=True)

#Analyze the activations for different types
analyze_type_attention(enc_dec_attention, seqs, true_types, pred_types,pred_annotations, attention_dir)
pdb.set_trace()
