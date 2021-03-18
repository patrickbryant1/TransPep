#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter
import logomaker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze attention.''')
parser.add_argument('--attention_dir', nargs=1, type= str, default=sys.stdin, help = '''path to attention.''')

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

def get_kingdom_attention(seqs, true_types, true_annotations, pred_types,pred_annotations,pred_annotation_probs, enc_dec_attention, attention_dir, types, kingdom):
    '''Analyze the attention for a certain kingdom
    '''

    AMINO_ACIDS = { 'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,
                    'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,
                    'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,
                    'Y':18,'V':19,'X':20
                  }
    annotation_conversion = {'S':0,'T':1,'L':2,'I':3,'M':4,'O':5}
    annotation_type_conversion = {'Sec/SPI': 0, 'Tat/SPI': 1, 'Sec/SPII': 2}

    # create color scheme
    annotation_color_scheme = {'S' : 'tab:blue','T' : 'tab:pink', 'L' : 'tab:purple',
                                'I': 'gray', 'M': 'k', 'O':'tab:gray'}

    for type in types:
        type_P = np.argwhere(true_types==types[type])
        type_pred_P = np.argwhere(pred_types==types[type])
        type_TP = np.intersect1d(type_P,type_pred_P)

        type_enc_dec_attention = enc_dec_attention[type_TP]
        type_seqs = seqs[type_TP]
        type_annotations = pred_annotations[type_TP]

        if type!='NO_SP':
            #Get all true positive CSs
            P_annotations = true_annotations[type_TP]
            P_CS = []
            for i in range(len(P_annotations)):
                P_CS.append(np.argwhere(P_annotations[i]==annotation_type_conversion[type])[-1,0])

            #Order the attention matrix properly
            ordered_type_enc_dec_attention = np.zeros((len(type_enc_dec_attention),40,40))
            for i in range(len(type_enc_dec_attention)):
                pdb.set_trace()
                #Upper left
                ordered_type_enc_dec_attention[i,:20,:20]=type_enc_dec_attention[i,P_CS[i]-20:P_CS[i],P_CS[i]-20:P_CS[i]]
                #Upper right
                ordered_type_enc_dec_attention[i,:20,20:]=type_enc_dec_attention[i,P_CS[i]-20:P_CS[i],P_CS[i]:P_CS[i]+20]
                #Lower left
                ordered_type_enc_dec_attention[i,20:,:20]=type_enc_dec_attention[i,P_CS[i]:P_CS[i]+20,P_CS[i]-20:P_CS[i]]
                #Lower right
                ordered_type_enc_dec_attention[i,20:,20:]=type_enc_dec_attention[i,P_CS[i]:P_CS[i]+20,P_CS[i]:P_CS[i]+20]

                pdb.set_trace()
        else:
            continue
        #Plot activation matrix around the CS
        fig,ax = plt.subplots(figsize=(9,9))
        im = plt.imshow(np.average(type_enc_dec_attention,axis=0))#In seqs on x, out annotations on y
        plt.xlabel('Sequence position')
        plt.ylabel('Annotation position')
        plt.title(type)
        plt.tight_layout()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.tight_layout()
        plt.savefig(attention_dir+kingdom+'_enc_dec_attention_'+str(types[type])+'.png',format='png',dpi=300)
        plt.close()


        #Get aa attention
        aa_attention = np.zeros((70,21))
        for i in range(70):
            col = type_seqs[:,i]
            #Go through all amino acids
            for aa in range(21):
                if aa not in col:
                    continue
                #Where col==aa
                aa_col_pos = np.argwhere(col==aa)
                #Get corresponding enc-dec attention
                aa_col_attention = np.max(type_enc_dec_attention[aa_col_pos,:,i]) #axis 0 = row in np, 1=col
                aa_attention[i,aa]=aa_col_attention

        #Get annotation attention
        annotation_attention = np.zeros((70,6))
        for j in range(70):
            row = type_annotations[:,j]
            for at in range(6):
                if at not in row:
                    continue
                #Where row==at
                at_row_pos = np.argwhere(row==at)
                #Get corresponding enc-dec attention
                at_row_attention = np.max(type_enc_dec_attention[at_row_pos,j,:]) #axis 0 = row in np, 1=col
                annotation_attention[j,at]= at_row_attention


        #Convert to dfs
        aa_attention_df = pd.DataFrame(aa_attention,columns = [*AMINO_ACIDS.keys()])
        annotation_attention = annotation_attention[::-1,:]
        annotation_attention_df = pd.DataFrame(annotation_attention,columns = [*annotation_conversion.keys()])

        #Logos
        #aa
        fig,ax = plt.subplots(figsize=(9/2.54,4.5/2.54))
        logomaker.Logo(aa_attention_df, color_scheme='skylign_protein')
        plt.ylabel('log2 Attention')
        plt.xticks([])
        plt.savefig(attention_dir+kingdom+'_aa_enc_dec_attention_logo_'+str(types[type])+'.png',format='png',dpi=300)
        plt.close()
        #annotation
        fig,ax = plt.subplots(figsize=(9/2.54,4.5/2.54))
        logomaker.Logo(annotation_attention_df, color_scheme=annotation_color_scheme)
        plt.xticks([])
        plt.ylabel('log2 Attention')
        plt.savefig(attention_dir+kingdom+'_annotation_enc_dec_attention_logo_'+str(types[type])+'.png',format='png',dpi=300)
        plt.close()

def analyze_attention(seqs, kingdoms, true_types, true_annotations, pred_types,pred_annotations,pred_annotation_probs, enc_dec_attention, attention_dir):
    '''Analyze the activations per type
    {'NO_SP':0,'SP':1,'TAT':2,'LIPO':3}
    SP: Sec substrates cleaved by SPase I (Sec/SPI),
    LIPO: Sec substrates cleaved by SPase II (Sec/SPII),
    TAT: Tat substrates cleaved by SPase I (Tat/SPI).
    annotation [S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]
    '''


    types = {'NO_SP':0,'Sec/SPI':1,'Tat/SPI':2,'Sec/SPII':3}
    kingdom_conversion = {'ARCHAEA':0,'NEGATIVE':2,'POSITIVE':3,'EUKARYA':1}

    for key in kingdom_conversion:
        kingdom_indices = np.argwhere(kingdoms==kingdom_conversion[key])[:,0]
        if key=='EUKRAYA':
            types = {'NO_SP':0,'Sec/SPI':1}
        get_kingdom_attention(seqs[kingdom_indices], true_types[kingdom_indices], true_annotations[kingdom_indices], pred_types[kingdom_indices],
        pred_annotations[kingdom_indices],pred_annotation_probs[kingdom_indices], enc_dec_attention[kingdom_indices], attention_dir+key+'/', types, key)
        pdb.set_trace()



######################MAIN######################
plt.rcParams.update({'font.size': 7})
args = parser.parse_args()
attention_dir = args.attention_dir[0]


#Parse
enc_attention = []
enc_dec_attention = []
pred_annotations = []
pred_annotation_probs = []
true_annotations = []
true_types = []
seqs = []
kingdoms = []
for test_partition in range(5):
    #Get true annotations and types
    true_annotations.extend([*np.load(attention_dir+'annotations_'+str(test_partition)+'.npy',allow_pickle=True)])
    true_types.extend([*np.load(attention_dir+'types_'+str(test_partition)+'.npy',allow_pickle=True)])
    #Get seqs
    seqs.extend([*np.load(attention_dir+'seqs_'+str(test_partition)+'.npy',allow_pickle=True)])
    #Kingdoms
    kingdoms.extend([*np.load(attention_dir+'kingdoms_'+str(test_partition)+'.npy',allow_pickle=True)])
    #For parition
    partition_enc_attention = []
    partition_enc_dec_attention = []
    partition_pred_annotations = []
    for valid_partition in np.setdiff1d(np.arange(5),test_partition):
        #Load
        partition_enc_attention.append(np.load(attention_dir+'enc_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))
        partition_enc_dec_attention.append(np.load(attention_dir+'enc_dec_attention_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))
        partition_pred_annotations.append(np.load(attention_dir+'pred_annotations_'+str(test_partition)+'_'+str(valid_partition)+'.npy',allow_pickle=True))

    #Array conversion
    partition_enc_attention = np.array(partition_enc_attention)
    partition_enc_dec_attention = np.array(partition_enc_dec_attention)
    partition_pred_annotations = np.array(partition_pred_annotations)
    #Average across validation splits
    print(partition_enc_dec_attention.shape, partition_enc_attention.shape)
    partition_enc_attention = np.average(partition_enc_attention,axis=0)
    partition_enc_dec_attention = np.average(partition_enc_dec_attention,axis=0)
    partition_pred_annotations = np.average(partition_pred_annotations,axis=0)

    #Append
    enc_attention.extend([*partition_enc_attention])
    enc_dec_attention.extend([*partition_enc_dec_attention])
    pred_annotation_probs.extend([*partition_pred_annotations])
    pred_annotations.extend([*np.argmax(partition_pred_annotations,axis=2)])

#Array conversion
enc_attention = np.array(enc_attention)
enc_dec_attention = np.array(enc_dec_attention)
pred_annotations = np.array(pred_annotations)
pred_annotation_probs = np.array(pred_annotation_probs)
true_annotations = np.array(true_annotations)
true_types = np.array(true_types)
seqs = np.array(seqs)
kingdoms = np.array(kingdoms)
kingdoms = np.argmax(kingdoms[:,0],axis=1)

#Get types
pred_types = get_pred_types(pred_annotations)
#Max across attention heads
enc_attention = np.max(enc_attention,axis=1)
enc_dec_attention = np.max(enc_dec_attention,axis=1)

#Analyze the activations for different types
analyze_attention(seqs, kingdoms, true_types, true_annotations, pred_types,pred_annotations,pred_annotation_probs, enc_dec_attention, attention_dir)
pdb.set_trace()
