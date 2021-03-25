#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import seaborn as sns
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


def pred_prob_vs_precision(type_probs_TP, type_probs_FP,type_index,kingdom,type,outname1, outname2):
    '''Compare the prediction probability of the TP and FP across the signal peptide region.
    '''

    if type_index!=3: #3=NO_SP
        TP_activation = np.sum(type_probs_TP[:,:,type_index],axis=1)
        FP_activation = np.sum(type_probs_FP[:,:,type_index],axis=1)
    else:
        TP_activation = np.sum(np.sum(type_probs_TP[:,:,type_index:],axis=1),axis=1)
        FP_activation = np.sum(np.sum(type_probs_FP[:,:,type_index:],axis=1),axis=1)

    #Distribution
    fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))
    sns.distplot(TP_activation,label='TP')
    sns.distplot(FP_activation, label='FP')
    plt.title(kingdom+' '+type)
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('Probability sum')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(outname1,format='png',dpi=300)
    plt.close()

    #Precision vs probability cutoff
    all_probs = np.concatenate([TP_activation,FP_activation])
    precision = []
    percent_TP = []
    for i in np.arange(0,max(all_probs),1):
        TP_above = len(np.argwhere(TP_activation>=i))
        FP_above = len(np.argwhere(FP_activation>=i))
        precision.append(TP_above/(TP_above+FP_above))
        percent_TP.append(100*TP_above/len(TP_activation))


    fig,ax1 = plt.subplots(figsize=(6/2.54,4.5/2.54))
    ax2 = ax1.twinx()
    ax1.plot(np.arange(0,max(all_probs),1),precision,color='tab:blue')
    ax2.plot(np.arange(0,max(all_probs),1),percent_TP,color='mediumseagreen')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_color('mediumseagreen')
    ax2.spines['left'].set_color('tab:blue')
    ax1.set_ylim([0,1.1])
    #Where is the precision 99%
    try:
        prec_cutoff = np.argwhere(np.array(precision)>=0.99)[0][0]
        print(kingdom,type,prec_cutoff,percent_TP[prec_cutoff])
        ax1.axvline(x=prec_cutoff,ymin=0,ymax=1,ls='--',color='gray',linewidth=1)
    except:
        print('No precision above 0.99')

    ax1.set_xlabel('Prob. sum cutoff')
    ax1.set_ylabel('Precision',color='tab:blue')
    ax2.set_ylabel('%TP',color='mediumseagreen')
    plt.title(kingdom+' '+type)
    plt.tight_layout()
    plt.savefig(outname2,format='png',dpi=300)
    plt.close()


def plot_attention_matrix(attention_matrix,type,kingdom,outname,figsize):
    '''Plot the encoder-decoder matrix
    '''
    #Plot activation matrix around the CS/or the whole matrix if no CS for the TP
    fig,ax = plt.subplots(figsize=figsize)
    im = plt.imshow(np.average(attention_matrix,axis=0),cmap='viridis') #In seqs on x, out annotations on y
    if attention_matrix.shape[2]==40:
        plt.axvline(19.5, color='y', linewidth=1, linestyle=':')
        plt.axhline(4.5, color='y', linewidth=1, linestyle=':')
        plt.xticks(ticks=np.arange(attention_matrix.shape[2]),labels=[-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        plt.yticks(ticks=np.arange(attention_matrix.shape[1]),labels=[-5,-4,-3,-2,-1,1,2,3,4,5])
    plt.xlabel('Sequence position')
    plt.ylabel('Annotation position')
    if kingdom == 'NEGATIVE':
        kingdom = 'Gram-negative bacteria'
    if kingdom == 'POSITIVE':
        kingdom = 'Gram-positive bacteria'
    plt.title(kingdom+' '+type)
    plt.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(outname,format='png',dpi=300)
    plt.close()

def convert_TP_seqs(type_seqs_TP):
    '''Convert the TP sequences back to AA to build a logo
    '''

    #Get df for logo
    aa_freqs = np.zeros((type_seqs_TP.shape[1],21))
    #Go through all cols:
    for i in range(type_seqs_TP.shape[1]):
        col = type_seqs_TP[:,i] #These have to be ordered just like the attention around the CS
                                #Make sure this has been done!
        #Go through all amino acids
        for aa in range(21):
            if aa not in col:
                continue
            else:
                #Where col==aa
                aa_col_pos = np.argwhere(col==aa)
                #Get freq
                aa_freqs[i,aa]=aa_col_pos.shape[0]/len(col)


    return aa_freqs


def get_kingdom_attention(seqs, true_types, true_annotations, pred_types,pred_annotations,pred_annotation_probs, enc_dec_attention, attention_dir, types, kingdom):
    '''Analyze the attention for a certain kingdom
    '''

    AMINO_ACIDS = { 'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,
                    'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,
                    'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,
                    'Y':18,'V':19,'X':20
                  }
    annotation_conversion = {'S':0,'T':1,'L':2,'I':3,'M':4,'O':5}
    annotation_type_conversion = {'Sec/SPI': 0, 'Tat/SPI': 1, 'Sec/SPII': 2, 'NO_SP':3}

    # create color scheme
    annotation_color_scheme = {'S' : 'tab:blue','T' : 'tab:pink', 'L' : 'tab:purple',
                                'I': 'gray', 'M': 'k', 'O':'tab:gray'}

    hp = 'tab:gray' #hydrophobic color
    small = 'mediumseagreen'
    polar = 'darkgray'
    neg = 'magenta'
    pos = 'royalblue'
    AA_color_scheme = { 'A':hp,'R':pos,'N':polar,'D':neg,'C':polar,'E':neg,
                    'Q':polar,'G':small,'H':pos,'I':hp,'L':hp,'K':pos,
                    'M':hp,'F':hp,'P':hp,'S':polar,'T':polar,'W':hp,
                    'Y':hp,'V':hp,'X':'k'
                  }
    #Go through all types
    for type in types:
        figsize=(9,9)
        type_P = np.argwhere(true_types==types[type])
        type_pred_P = np.argwhere(pred_types==types[type])[:,0]
        type_TP = np.intersect1d(type_P,type_pred_P)
        type_FP = np.setdiff1d(type_pred_P,type_TP)
        #Attention of pred pos
        type_enc_dec_attention = enc_dec_attention[type_pred_P]
        #TP
        type_enc_dec_attention_TP = enc_dec_attention[type_TP]
        #seqs and annotations of TP
        type_seqs_TP = seqs[type_TP]


        type_annotations_TP = pred_annotations[type_TP]
        #Probabilities
        type_probs_TP = pred_annotation_probs[type_TP]
        type_probs_FP = pred_annotation_probs[type_FP]

        #Plot type probabilities
        #pred_prob_vs_precision(type_probs_TP, type_probs_FP,annotation_type_conversion[type],kingdom, type ,attention_dir+kingdom+'_type_prob'+str(types[type])+'.png',attention_dir+kingdom+'_type_precision'+str(types[type])+'.png')


        if type!='NO_SP':

            #Get all positive CSs that have TP type
            P_annotations = true_annotations[type_TP]
            P_CS = []
            for i in range(len(P_annotations)):
                P_CS.append(np.argwhere(P_annotations[i]==annotation_type_conversion[type])[-1,0])
            P_CS = np.array(P_CS)
            #Get all pred positive CSs from the true positives (all the other will be wrong)
            P_CS_pred = []
            P_annotations_pred = type_annotations_TP
            for i in range(len(P_annotations_pred)):
                P_CS_pred.append(np.argwhere(P_annotations_pred[i]==annotation_type_conversion[type])[-1,0])
            P_CS_pred = np.array(P_CS_pred)

            #Get TP CS
            CS_diff = P_CS-P_CS_pred
            CS_TP = np.argwhere(np.absolute(CS_diff)<=3)[:,0]
            #Get the mapping to the type TPs
            CS_TP =type_TP[CS_TP]
            CS_FP = np.setdiff1d(type_pred_P,CS_TP)

            #Order the attention matrix and seqs around the CS properly
            ordered_type_enc_dec_attention_TP = np.zeros((len(type_enc_dec_attention_TP),10,40))
            ordered_type_seqs_TP = np.zeros((len(type_seqs_TP),40))

            for i in range(len(type_enc_dec_attention_TP)):

                #Upper left
                ul = max(P_CS[i]-19,1) #The CS is the last position with SP. The cleavages will thus happen directly after this position
                ul_len = min(P_CS[i],20)
                ordered_type_enc_dec_attention_TP[i,:10,20-ul_len:20]=type_enc_dec_attention_TP[i,P_CS[i]-4:P_CS[i]+6,ul:P_CS[i]+1]
                ordered_type_seqs_TP[i,20-ul_len:20]=type_seqs_TP[i,ul:P_CS[i]+1]

                #Upper right
                ur = min(P_CS[i]+1+20,70) #P_CS is the last position with SP
                ur_len = min(70-(P_CS[i]+1),20) #P_CS[i]+1, since zero indexed
                ordered_type_enc_dec_attention_TP[i,:10,40-ur_len:]=type_enc_dec_attention_TP[i,P_CS[i]-4:P_CS[i]+6,P_CS[i]+1:ur]
                ordered_type_seqs_TP[i,40-ur_len:]=type_seqs_TP[i,P_CS[i]+1:ur]

            #Reassign
            type_enc_dec_attention_TP = ordered_type_enc_dec_attention_TP
            type_seqs_TP = ordered_type_seqs_TP
            figsize=(9,4.5)

            #Plot attention matrix
            #TP
            plot_attention_matrix(ordered_type_enc_dec_attention_TP,type,kingdom,attention_dir+kingdom+'_enc_dec_attention_'+str(types[type])+'_TP_CS_area.png',figsize)

            #Convert and save the sequences to build a logo
            aa_freqs_type_TP = convert_TP_seqs(type_seqs_TP)
            #Convert to df
            aa_seq_df = pd.DataFrame(aa_freqs_type_TP,columns = [*AMINO_ACIDS.keys()])
            #Transform
            aa_seq_df =logomaker.transform_matrix(aa_seq_df,from_type='probability',to_type='information')

            #Logo
            fig,ax = plt.subplots(figsize=(figsize[0]/2.54,figsize[1]/2.54))
            aa_logo = logomaker.Logo(aa_seq_df, color_scheme=AA_color_scheme)
            plt.ylabel('log2 Frequency')
            plt.xticks([])
            aa_logo.ax.axvline(19.5, color='k', linewidth=2, linestyle=':')
            plt.savefig(attention_dir+kingdom+'_aa_seq_logo_'+str(types[type])+'.png',format='png',dpi=300)
            plt.close()

        continue

        #Plot attention matrix
        #TP
        plot_attention_matrix(type_enc_dec_attention[np.argwhere(np.isin(type_pred_P,type_TP))[:,0]],type,kingdom,attention_dir+kingdom+'_enc_dec_attention_'+str(types[type])+'_TP.png',(9/2.54,9/2.54))
        #FP
        #plot_attention_matrix(type_enc_dec_attention[np.argwhere(np.isin(type_pred_P,type_FP))[:,0]],type,kingdom,attention_dir+kingdom+'_enc_dec_attention_'+str(types[type])+'_FP.png',(9/2.54,9/2.54))

        #Get aa attention
        aa_attention = np.zeros((type_enc_dec_attention_TP.shape[2],21))
        for i in range(len(aa_attention)):
            col = type_seqs_TP[:,i] #These have to be ordered just like the attention around the CS
            #Go through all amino acids
            for aa in range(21):
                if aa not in col:
                    continue
                else:
                    #Where col==aa
                    aa_col_pos = np.argwhere(col==aa)
                    #Get corresponding enc-dec attention
                    aa_col_attention = np.average(type_enc_dec_attention_TP[aa_col_pos,:,i]) #axis 0 = row in np, 1=col
                    aa_attention[i,aa]=aa_col_attention

        #Get annotation attention
        annotation_attention = np.zeros((type_enc_dec_attention_TP.shape[1],6))
        for j in range(len(annotation_attention)):
            row = type_annotations_TP[:,j]
            for at in range(6):
                if at not in row:
                    continue
                else:
                    #Where row==at
                    at_row_pos = np.argwhere(row==at)
                    #Get corresponding enc-dec attention
                    at_row_attention = np.average(type_enc_dec_attention_TP[at_row_pos,j,:]) #axis 0 = row in np, 1=col
                    annotation_attention[j,at]= at_row_attention


        #Convert to dfs
        aa_attention_df = pd.DataFrame(aa_attention,columns = [*AMINO_ACIDS.keys()])
        annotation_attention = annotation_attention[::-1,:]
        annotation_attention_df = pd.DataFrame(annotation_attention,columns = [*annotation_conversion.keys()])
        #Logos
        #aa
        fig,ax = plt.subplots(figsize=(figsize[0]/2.54,figsize[1]/2.54))
        aa_logo = logomaker.Logo(aa_attention_df, color_scheme=AA_color_scheme)
        plt.ylabel('log2 Attention')
        plt.xticks([])
        if type!='NO_SP':
            aa_logo.ax.axvline(19.5, color='k', linewidth=2, linestyle=':')
        plt.savefig(attention_dir+kingdom+'_aa_enc_dec_attention_logo_'+str(types[type])+'.png',format='png',dpi=300)
        plt.close()
        #annotation
        if type!='NO_SP':
            fig,ax = plt.subplots(1,1,figsize=[2.5,2.5])
            annotation_logo = logomaker.Logo(annotation_attention_df,ax=ax, color_scheme=annotation_color_scheme)
            annotation_logo.ax.axvline(4.5, color='k', linewidth=2, linestyle=':')
        else:
            fig,ax = plt.subplots(figsize=(figsize[0]/2.54,figsize[1]/2.54))
            annotation_logo = logomaker.Logo(annotation_attention_df, color_scheme=annotation_color_scheme)

        plt.xticks([])
        plt.ylabel('log2 Attention')
        annotation_logo.fig.tight_layout()
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
