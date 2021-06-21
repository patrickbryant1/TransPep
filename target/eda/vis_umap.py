#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
#sys.path.insert(0, "../../")
import numpy as np
import pandas as pd
import time

import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Visualuze the UMMAP embeddings of raw sequences from DeepLoc''')
parser.add_argument('--fasta', nargs=1, type= str, default=sys.stdin, help = 'Path to raw sequences.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')

def get_fasta(filename):
    '''Read in a fasta file
    '''
    AMINO_ACIDS = { 'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,
                'Q':7,'G':8,'H':9,'I':10,'L':11,'K':12,
                'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,
                'Y':19,'V':20,'X':21,'U':21, 'Z':21, 'B':21
              }
    IDs = []
    Seqs = []
    Lens = []
    Types = []


    #Go through each line
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            if line[0]=='>': #Id
                line = line.split()
                IDs.append(line[0][1:])
                Types.append(line[1][:-2])

            else: #Sequence
                current_seq = []
                for char in line:
                    current_seq.append(AMINO_ACIDS[char])
                #Seqlen
                Lens.append(len(current_seq))
                #Pad or cut
                if len(current_seq)>1000:
                    current_seq = current_seq[:1000]
                else:
                    pad = np.zeros(1000)
                    pad[:len(current_seq)]=current_seq
                    current_seq = pad
                #Seq
                Seqs.append(current_seq)



    fasta_df = pd.DataFrame()
    fasta_df['ID'] = IDs
    fasta_df['Type'] = Types
    fasta_df['Sequence'] = Seqs
    fasta_df['Seqlen'] = Lens
    pdb.set_trace()



def vis_umap(us, all_types):
    '''Visualize UMAP
    '''
    #5 classes of transit peptides

    #Colors
    type_descr = {1:'NA',2:'SP',3:'MT',4:'CH',5:'TH'}
    type_colors = {'NA':'grey','SP':'tab:blue','MT':'magenta','CH':'g','TH':'k'}
    types = [type_descr[i] for i in all_types]
    org_descr = {0:'No plant',1:'Plant'}
    org_colors = {'No plant':'grey','Plant':'g'}
    orgs = [org_descr[i] for i in all_orgs]
    df = pd.DataFrame()
    df['u1']=us[:,0]
    df['u2']=us[:,1]
    df['u1_seq']=us_seq[:,0]
    df['u2_seq']=us_seq[:,1]
    df['type']=types
    df['org']=orgs


    #Types seq
    fig,ax = plt.subplots(figsize=(12/2.54,12/2.54))
    for type in df.type.unique():
        sel = df[df.type==type]
        plt.scatter(sel['u1_seq'],sel['u2_seq'],s=2,color=type_colors[type],label=type,alpha=0.5)
    plt.legend()
    plt.title('Sequences')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('u1')
    plt.ylabel('u2')
    plt.tight_layout()
    plt.savefig(outdir+'type_umap_seq'+suffix+'.png',dpi=300)


######################MAIN######################
args = parser.parse_args()
#Set font size
matplotlib.rcParams.update({'font.size': 7})
fasta = args.fasta[0]
outdir = args.outdir[0]

#Get seqs
fasta_df = get_fasta(fasta)

#Get seq and emb projections
import umap
print('Mapping UMAP for encodings...')
us = umap.UMAP().fit_transform(all_encodings_z)

#save
np.save(datadir+'umap'+str(i)+'_'+str(j)+'.npy',us)
np.save(datadir+'umap_seq'+str(i)+'_'+str(j)+'.npy',us_seq)

#Visualize
vis_umap(us,us_seq, all_types, all_orgs, str(i)+'_'+str(j))
