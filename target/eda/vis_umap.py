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
    Solubility = []


    #Go through each line
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            if line[0]=='>': #Id
                line = line.split()
                IDs.append(line[0][1:])
                Types.append(line[1][:-2])
                Solubility.append(line[1][-1])

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
    fasta_df['Solubility'] = Solubility
    fasta_df['Sequence'] = Seqs
    fasta_df['Seqlen'] = Lens

    Seqs = fasta_df.Sequence.values
    return fasta_df, np.array([*Seqs])



def vis_umap(us, all_types, outdir):
    '''Visualize UMAP
    '''
    #5 classes of transit peptides

    #Colors
    type_colors = {'Cell.membrane':'grey', 'Cytoplasm-Nucleus':'tab:blue', 'Cytoplasm':'orange',
       'Endoplasmic.reticulum':'r', 'Golgi.apparatus':'magenta', 'Lysosome/Vacuole':'g',
       'Mitochondrion':'tab:purple', 'Nucleus':'k', 'Peroxisome':'royalblue', 'Plastid':'mediumseagreen',
       'Extracellular':'maroon'}

    df = pd.DataFrame()
    df['u1']=us[:,0]
    df['u2']=us[:,1]
    df['type']=all_types

    #Types seq
    fig,ax = plt.subplots(figsize=(12/2.54,12/2.54))
    for type in df.type.unique():
        sel = df[df.type==type]
        plt.scatter(sel['u1'],sel['u2'],s=2,color=type_colors[type],label=type,alpha=0.5)
    plt.legend()
    plt.title('Sequences')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('u1')
    plt.ylabel('u2')
    plt.tight_layout()
    plt.savefig(outdir+'type_umap_seq.png',dpi=300)


######################MAIN######################
args = parser.parse_args()
#Set font size
matplotlib.rcParams.update({'font.size': 7})
fasta = args.fasta[0]
outdir = args.outdir[0]

#Get seqs
fasta_df, Seqs = get_fasta(fasta)
fasta_df
#Get seq and emb projections
# import umap
# print('Mapping UMAP for encodings...')
# us = umap.UMAP().fit_transform(Seqs)
##save
#np.save(outdir+'deeploc_umap.npy',us)

from sklearn.manifold import TSNE
us = TSNE(n_components=2).fit_transform(Seqs)
#Visualize
vis_umap(us, fasta_df.Type.values, outdir)
