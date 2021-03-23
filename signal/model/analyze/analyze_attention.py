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

def plot_attention_distribution(aa_area, attention_area, type_pred_P, type_TP, type_FP, kingdom, type, outname):
    '''Plot the attention vs amino acid area
    '''
    #Plot
    fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))

    for i in range(len(aa_area)):
        if type_pred_P[i] in type_TP:
            color = 'b'
            alpha=0.2
        else:
            color='r'
            alpha=0.5
        plt.plot(aa_area[i],attention_area[i],color=color,alpha=alpha,linewidth=1)

    plt.title(kingdom+' '+type)
    plt.xlabel('Number of columns')
    plt.ylabel('% Attention')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(outname,format='png',dpi=300)
    plt.close()


def get_attention_distribution(attention_matrix):
    '''Check how the distribution of the total attention varies btw TP and FP.
    '''

    n_cols = []
    fetched_attention = []
    for i in range(len(attention_matrix)):
        sample = attention_matrix[i]
        total_sample_attention = np.sum(sample)
        col_sums = np.sum(sample,axis=0)
        max_col = np.argmax(col_sums)

        #Search the area around the max attention and see how far away you have to go to obtain 90 % of the attention
        fetched_sample_attention = [col_sums[max_col]/total_sample_attention]
        n_sample_cols = [1]
        m=1 #minus change
        p=1 #plus change
        if max_col-m<0:
            p+=1
        if max_col+p>len(col_sums):
            m+=1
        li = max(max_col-m,0) #Left index
        ri = min(max_col+p,len(col_sums)) #Right index

        while li>0 or ri<len(col_sums):
            fetched_sample_attention.append(np.sum(col_sums[li:ri])/total_sample_attention)
            n_sample_cols.append(ri+1-li)
            if max_col-m<=0:
                p+=1
            if max_col+p>=len(col_sums):
                m+=1
            #Increase index
            m+=1
            p+=1
            li = max(max_col-m,0) #Left index
            ri = min(max_col+p,len(col_sums)) #Right index

            if n_sample_cols[-1]-n_sample_cols[-2]!=2:
                pdb.set_trace()
        n_cols.append(np.array(n_sample_cols))
        fetched_attention.append(np.array(fetched_sample_attention))

    return np.array(n_cols), np.array(fetched_attention)

def calc_best_percentage_split(aa_area, attention_area, type_TP,kingdom,type,outname):
    '''Go through all distances in steps of 2 aa and search for the best attention % cutoff.
    '''


    pos_included_95 = []
    percentage_95 = []
    pos_included_90 = []
    percentage_90 = []
    pos_included_85 = []
    percentage_85 = []

    for i in range(aa_area.shape[1]):
        attention_area_i = attention_area[:,i]
        perc_above_cutoff = [] #Percent of all curves above the cutoff
        precision_above_cutoff = [] #Percent TP above cutoff
        for p in np.arange(min(attention_area_i),max(attention_area_i),0.01):
            #Get above cutoff
            above_cutoff = np.argwhere(attention_area_i>=p)[:,0]
            if len(above_cutoff)<1:
                continue

            #Get overlap with TP
            n_overlap = np.intersect1d(above_cutoff,type_TP).shape[0]
            precision_above_cutoff.append(n_overlap/len(above_cutoff))

            #How many TP of all TP above cutoff?
            perc_above_cutoff.append(n_overlap/len(type_TP))

        #Plot
        #Get cutoff where precision is 95 %
        best_cutoff_pos = np.argwhere(np.array(precision_above_cutoff)>=0.95)
        if len(best_cutoff_pos)>0:
            percentage_95.append(perc_above_cutoff[best_cutoff_pos[0][0]])
            pos_included_95.append(aa_area[0,i])

        #Get cutoff where precision is 90 %
        best_cutoff_pos = np.argwhere(np.array(precision_above_cutoff)>=0.90)
        if len(best_cutoff_pos)>0:
            percentage_90.append(perc_above_cutoff[best_cutoff_pos[0][0]])
            pos_included_90.append(aa_area[0,i])

        #Get cutoff where precision is 85 %
        best_cutoff_pos = np.argwhere(np.array(precision_above_cutoff)>=0.85)
        if len(best_cutoff_pos)>0:
            percentage_85.append(perc_above_cutoff[best_cutoff_pos[0][0]])
            pos_included_85.append(aa_area[0,i])

    #Plot
    fig,ax = plt.subplots(figsize=(4.5/2.54,4.5/2.54))
    plt.plot(pos_included_95,np.array(percentage_95)*100,label='95%',color='tab:blue',alpha=0.5)
    plt.plot(pos_included_90,np.array(percentage_90)*100, label='90%',color='tab:green',alpha=0.5)
    plt.plot(pos_included_85,np.array(percentage_85)*100, label='85%',color='tab:gray',alpha=0.5)
    plt.legend(title='Precision')
    plt.xlabel('Number of columns',fontsize=7)
    plt.ylabel('% TP selected',fontsize=7)
    plt.title(kingdom + ' ' +type)
    plt.ylim([0,110])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(outname,format='png',dpi=300)
    plt.close()


def pred_prob_vs_precision(type_probs_TP, type_probs_FP,type_index,kingdom,type,outname1, outname2):
    '''Compare the prediction probability of the TP and FP across the signal peptide region.
    '''

    if type!=3: #3=NO_SP
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
        percent_TP.append(TP_above/len(TP_activation))

    fig,ax1 = plt.subplots(figsize=(6/2.54,4.5/2.54))
    ax2 = ax1.twinx()
    ax1.plot(np.arange(0,max(all_probs),1),precision,color='tab:blue')
    ax2.plot(np.arange(0,max(all_probs),1),percent_TP,color='mediumseagreen')
    ax1.spines['top'].set_visible(False)
    plt.xlabel('Prob. sum cutoff')
    ax1.set_ylabel('Precision')
    ax2.set_ylabel('%TP')
    plt.title(kingdom+' '+type)
    plt.tight_layout()
    plt.savefig(outname2,format='png',dpi=300)
    plt.close()
    pdb.set_trace()

def plot_attention_matrix(attention_matrix,type,kingdom,outname,figsize):
    '''Plot the encoder-decoder matrix
    '''
    #Plot activation matrix around the CS/or the whole matrix if no CS for the TP
    fig,ax = plt.subplots(figsize=figsize)
    im = plt.imshow(np.average(attention_matrix,axis=0)) #In seqs on x, out annotations on y
    if attention_matrix.shape[2]==40:
        plt.axvline(19.5, color='y', linewidth=1, linestyle=':')
        plt.axhline(4.5, color='y', linewidth=1, linestyle=':')
        plt.xticks(ticks=np.arange(attention_matrix.shape[2]),labels=[-20,-19,-18,-17,-16,-15,-14,-13,-12,-11,-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        plt.yticks(ticks=np.arange(attention_matrix.shape[1]),labels=[-5,-4,-3,-2,-1,1,2,3,4,5])
    plt.xlabel('Sequence position')
    plt.ylabel('Annotation position')
    plt.title(kingdom+' '+type)
    plt.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()
    plt.savefig(outname,format='png',dpi=300)
    plt.close()


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
        pred_prob_vs_precision(type_probs_TP, type_probs_FP,annotation_type_conversion[type],kingdom, type ,attention_dir+kingdom+'_type_prob'+str(types[type])+'.png',attention_dir+kingdom+'_type_precision'+str(types[type])+'.png')
        continue
        # #Calculate the attention localization
        #aa_area, attention_area = get_attention_distribution(type_enc_dec_attention)
        #Plot
        #plot_attention_distribution(aa_area, attention_area, type_pred_P, type_TP, type_FP, kingdom, type, attention_dir+kingdom+'_attention_area_type'+str(types[type])+'.png')

        #Get the best percentage split
        #calc_best_percentage_split(aa_area, attention_area, np.argwhere(np.isin(type_pred_P,type_TP))[:,0],kingdom,type,attention_dir+kingdom+'_precision_type'+str(types[type])+'.png')

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

            #Get the calculated attention localization for the TP CSs
            #Plot
            #plot_attention_distribution(aa_area, attention_area, type_pred_P, CS_TP, CS_FP, kingdom, type+' CS', attention_dir+kingdom+'_attention_area_CS'+str(types[type])+'.png')

            #Get the best percentage split
            #calc_best_percentage_split(aa_area, attention_area, np.argwhere(np.isin(type_pred_P,CS_TP))[:,0],kingdom,type+' CS',attention_dir+kingdom+'_precision_CS'+str(types[type])+'.png')

            #Order the attention matrix properly
            ordered_type_enc_dec_attention_TP = np.zeros((len(type_enc_dec_attention_TP),10,40))
            for i in range(len(type_enc_dec_attention_TP)):

                #Upper left
                ul = max(P_CS[i]-20,0)
                ul_len = min(P_CS[i],20)
                ordered_type_enc_dec_attention_TP[i,:10,20-ul_len:20]=type_enc_dec_attention_TP[i,P_CS[i]-5:P_CS[i]+5,ul:P_CS[i]]

                #Upper right
                ur = min(P_CS[i]+20,70)
                ur_len = min(70-P_CS[i],20)
                ordered_type_enc_dec_attention_TP[i,:10,40-ur_len:]=type_enc_dec_attention_TP[i,P_CS[i]-5:P_CS[i]+5,P_CS[i]:ur]

            type_enc_dec_attention_TP = ordered_type_enc_dec_attention_TP
            figsize=(9,4.5)

            #Plot attention matrix
            #TP
            #plot_attention_matrix(ordered_type_enc_dec_attention_TP,type,kingdom,attention_dir+kingdom+'_enc_dec_attention_'+str(types[type])+'_TP_CS_area.png',figsize)



        #Plot attention matrix
        #TP
        #plot_attention_matrix(type_enc_dec_attention[np.argwhere(np.isin(type_pred_P,type_TP))[:,0]],type,kingdom,attention_dir+kingdom+'_enc_dec_attention_'+str(types[type])+'_TP.png',(9/2.54,9/2.54))
        #FP
        #plot_attention_matrix(type_enc_dec_attention[np.argwhere(np.isin(type_pred_P,type_FP))[:,0]],type,kingdom,attention_dir+kingdom+'_enc_dec_attention_'+str(types[type])+'_FP.png',(9/2.54,9/2.54))

        #Get aa attention
        aa_attention = np.zeros((type_enc_dec_attention_TP.shape[2],21))
        for i in range(len(aa_attention)):
            col = type_seqs_TP[:,i]
            #Go through all amino acids
            for aa in range(21):
                if aa not in col:
                    continue
                #Where col==aa
                aa_col_pos = np.argwhere(col==aa)
                #Get corresponding enc-dec attention
                aa_col_attention = np.max(type_enc_dec_attention_TP[aa_col_pos,:,i]) #axis 0 = row in np, 1=col
                aa_attention[i,aa]=aa_col_attention

        #Get annotation attention
        annotation_attention = np.zeros((type_enc_dec_attention_TP.shape[1],6))
        for j in range(len(annotation_attention)):
            row = type_annotations_TP[:,j]
            for at in range(6):
                if at not in row:
                    continue
                #Where row==at
                at_row_pos = np.argwhere(row==at)
                #Get corresponding enc-dec attention
                at_row_attention = np.max(type_enc_dec_attention_TP[at_row_pos,j,:]) #axis 0 = row in np, 1=col
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
        if type!='NO_SP':
            fig,ax = plt.subplots(1,1,figsize=[2.5,2.5])
            annotation_logo = logomaker.Logo(annotation_attention_df,ax=ax, color_scheme=annotation_color_scheme)
            annotation_logo.ax.axvline(4.5, color='k', linewidth=1, linestyle=':')
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
