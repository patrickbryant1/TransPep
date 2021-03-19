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

def precision_vs_attention(attention_matrix):
    '''Check how many aa are required to capture 90 % of the total attention
    when the type is correctly predicted.
    '''

    n_rows = []
    fetched_attention = []
    for i in range(len(attention_matrix)):
        sample = attention_matrix[i]
        total_sample_attention = np.sum(sample)
        row_sums = np.sum(sample,axis=0)
        max_row = np.argmax(row_sums)

        #Search the area around the max attention and see how far away you have to go to obtain 90 % of the attention
        fetched_sample_attention = [row_sums[max_row]/total_sample_attention]
        n_sample_rows = [1]
        m=1 #minus change
        p=1 #plus change
        li = max(max_row-m,0) #Left index
        ri = min(max_row+p,len(row_sums)) #Right index

        while li>0 or ri<len(row_sums):
            fetched_sample_attention.append(np.sum(row_sums[li:ri])/total_sample_attention)
            n_sample_rows.append(ri-li)
            m+=1
            p+=1
            if max_row-m<=0:
                p+=1
            if max_row+p>=len(row_sums):
                m+=1

            li = max(max_row-m,0) #Left index
            ri = min(max_row+p,len(row_sums)) #Right index

        n_rows.append(np.array(n_sample_rows))
        fetched_attention.append(np.array(fetched_sample_attention))

    return n_rows, fetched_attention

def calc_best_percentage_split(aa_area, attention_area, type_TP_or_not,kingdom,type):
    '''Go through all distances in steps of 2 aa and search for the best
    attention % cutoff.
    '''

    perc_above_cutoff = [] #Percent of all curves above the cutoff
    precision_above_cutoff = [] #Percent TP above cutoff
    best_attention_cutoff = [] #Best attention cutoff
    #Get only TP
    type_TP = np.argwhere(type_TP_or_not==1)[:,0]
    for i in range(aa_area.shape[1]):
        aa_area_i = aa_area[:,i]
        attention_area_i = attention_area[:,i]

        perc_above_cutoff = [] #Percent of all curves above the cutoff
        precision_above_cutoff = [] #Percent TP above cutoff
        for p in np.arange(min(attention_area_i),max(attention_area_i),0.01):
            #Get above cutoff
            above_cutoff = np.argwhere(attention_area_i>=p)[:,0]
            if len(above_cutoff)<1:
                continue
            perc_above_cutoff.append(len(above_cutoff)/len(aa_area))
            #Get overlap with TP
            n_overlap = np.intersect1d(above_cutoff,type_TP).shape[0]
            precision_above_cutoff.append(n_overlap/len(above_cutoff))
        #Plot
        plt.plot(perc_above_cutoff,precision_above_cutoff)
    plt.xlabel('% Selected')
    plt.ylabel('Precision')
    plt.title(kingom + ' ' +type)
    plt.ylim([0,1])
    plt.show()
    pdb.set_trace()



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


    all_aa_area = []
    all_attention_area = []

    for type in types:
        figsize=(9,9)
        type_P = np.argwhere(true_types==types[type])
        type_pred_P = np.argwhere(pred_types==types[type])
        type_TP = np.intersect1d(type_P,type_pred_P)

        type_enc_dec_attention = enc_dec_attention[type_TP]
        type_seqs = seqs[type_TP]
        type_annotations = pred_annotations[type_TP]

        #Look at the attention for the specific type
        aa_area, attention_area = precision_vs_attention(enc_dec_attention[type_pred_P[:,0]])
        all_aa_area.append(aa_area)
        all_attention_area.append(aa_area)
        #Plot
        fig,ax = plt.subplots(figsize=(9/2.54,9/2.54))
        type_TP_or_not = []
        for i in range(len(aa_area)):
            if type_pred_P[i][0] in type_TP:
                type_TP_or_not.append(1)
                color = 'b'
            else:
                type_TP_or_not.append(0)
                color='r'
            plt.plot(aa_area[i],attention_area[i],color=color,alpha=0.2)
        plt.title(kingdom+' '+type)
        plt.xlabel('Number of rows surrounding max attention')
        plt.ylabel('% Attention')
        plt.savefig(attention_dir+kingdom+'_attention_area_type'+str(types[type])+'.png',format='png',dpi=300)
        plt.close()



        if type!='NO_SP':
            #Calculate the best splitting point
            calc_best_percentage_split(np.array(aa_area), np.array(attention_area), np.array(type_TP_or_not),kingdom,type)

            #Get all positive CSs that have TP type
            P_annotations = true_annotations[type_TP]
            P_CS = []
            for i in range(len(P_annotations)):
                P_CS.append(np.argwhere(P_annotations[i]==annotation_type_conversion[type])[-1,0])
            P_CS = np.array(P_CS)
            #Get all pred positive CSs from the true positives (all the other will be wrong)
            P_CS_pred = []
            P_annotations_pred = type_annotations
            for i in range(len(P_annotations_pred)):
                P_CS_pred.append(np.argwhere(P_annotations_pred[i]==annotation_type_conversion[type])[-1,0])
            P_CS_pred = np.array(P_CS_pred)

            #Get TP CS
            CS_diff = P_CS-P_CS_pred
            CS_TP = np.argwhere(np.absolute(CS_diff)<=3)[:,0]
            #Get the mapping to the type TPs
            CS_TP =type_TP[CS_TP]

            #Plot CS attention
            fig,ax = plt.subplots(figsize=(9/2.54,9/2.54))
            for i in range(len(aa_area)):
                if type_pred_P[i][0] in CS_TP:
                    color = 'b'
                else:
                    color='r'
                plt.plot(aa_area[i],attention_area[i],color=color,alpha=0.2)
            plt.title(kingdom+' CS '+type)
            plt.xlabel('Number of rows surrounding max attention')
            plt.ylabel('% Attention')
            plt.savefig(attention_dir+kingdom+'_attention_area_CS'+str(types[type])+'.png',format='png',dpi=300)
            plt.close()
            continue

        else:
            continue

            #Order the attention matrix properly
            ordered_type_enc_dec_attention = np.zeros((len(type_enc_dec_attention),10,40))
            for i in range(len(type_enc_dec_attention)):

                #Upper left
                ul = max(P_CS[i]-20,0)
                ul_len = min(P_CS[i],20)
                ordered_type_enc_dec_attention[i,:10,20-ul_len:20]=type_enc_dec_attention[i,P_CS[i]-5:P_CS[i]+5,ul:P_CS[i]]

                #Upper right
                ur = min(P_CS[i]+20,70)
                ur_len = min(70-P_CS[i],20)
                ordered_type_enc_dec_attention[i,:10,40-ur_len:]=type_enc_dec_attention[i,P_CS[i]-5:P_CS[i]+5,P_CS[i]:ur]

            type_enc_dec_attention = ordered_type_enc_dec_attention
            figsize=(9,4.5)

        #Plot activation matrix around the CS
        fig,ax = plt.subplots(figsize=figsize)
        im = plt.imshow(np.average(type_enc_dec_attention,axis=0))#In seqs on x, out annotations on y
        if type!='NO_SP':
            plt.axvline(19.5, color='y', linewidth=1, linestyle=':')
            plt.axhline(4.5, color='y', linewidth=1, linestyle=':')
            plt.xticks(ticks=np.arange(type_enc_dec_attention.shape[2]),labels=[-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
            plt.yticks(ticks=np.arange(type_enc_dec_attention.shape[1]),labels=[-5,-4,-3,-2,-1,1,2,3,4,5])
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
        aa_attention = np.zeros((type_enc_dec_attention.shape[2],21))
        for i in range(len(aa_attention)):
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
        annotation_attention = np.zeros((type_enc_dec_attention.shape[1],6))
        for j in range(len(annotation_attention)):
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
        fig,ax = plt.subplots(figsize=(figsize[0]/2.54,figsize[1]/2.54))
        aa_logo = logomaker.Logo(aa_attention_df, color_scheme='skylign_protein')
        plt.ylabel('log2 Attention')
        plt.xticks([])
        if type!='NO_SP':
            aa_logo.ax.axvline(19.5, color='k', linewidth=1, linestyle=':')
        plt.savefig(attention_dir+kingdom+'_aa_enc_dec_attention_logo_'+str(types[type])+'.png',format='png',dpi=300)
        plt.close()
        #annotation
        fig,ax = plt.subplots(figsize=(figsize[1]/2.54,figsize[1]/2.54))
        attention_logo = logomaker.Logo(annotation_attention_df, color_scheme=annotation_color_scheme)
        if type!='NO_SP':
            attention_logo.ax.axvline(4.5, color='k', linewidth=1, linestyle=':')
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
        if key=='EUKARYA':
            types = {'NO_SP':0,'Sec/SPI':1}
        get_kingdom_attention(seqs[kingdom_indices], true_types[kingdom_indices], true_annotations[kingdom_indices], pred_types[kingdom_indices],
        pred_annotations[kingdom_indices],pred_annotation_probs[kingdom_indices], enc_dec_attention[kingdom_indices], attention_dir+key+'/', types, key)




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
