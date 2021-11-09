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

sys.path.insert(0, "../")
from process_data import parse_and_format

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze benchmark sequences.''')
parser.add_argument('--fasta', nargs=1, type= str, default=sys.stdin, help = '''path to fasta.''')

#####################FUNCTIONS#####################

def get_freqs(seqs):
    '''Convert the seqs to freqs
    '''

    #Get df for logo
    aa_freqs = np.zeros((seqs.shape[1],21))
    #Go through all cols:
    for i in range(seqs.shape[1]):
        col = seqs[:,i] #These have to be ordered just like the attention around the CS
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


######################MAIN######################
plt.rcParams.update({'font.size': 7})
args = parser.parse_args()
fasta = args.fasta[0]

data, Seqs, Annotations = parse_and_format(fasta)

AMINO_ACIDS = { 'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,
                'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,
                'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,
                'Y':18,'V':19,'X':20
              }
#Color scheme
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
for type in data.Type.unique():
    type_data = data[data.Type==type]
    type_Seqs = Seqs[type_data.index]
    #Get freqs
    aa_freqs = get_freqs(type_Seqs)
    #Convert to df
    aa_seq_df = pd.DataFrame(aa_freqs,columns = [*AMINO_ACIDS.keys()])
    #Transform
    aa_seq_df =logomaker.transform_matrix(aa_seq_df,from_type='probability',to_type='information')

    #Logo
    fig,ax = plt.subplots(figsize=(9/2.54,9/2.54))
    aa_logo = logomaker.Logo(aa_seq_df, color_scheme=AA_color_scheme)
    plt.ylabel('Information')
    plt.xlabel('Position')
    plt.title(type)
    plt.tight_layout()
    plt.savefig('bench_seq_logo_'+type+'.png',format='png',dpi=300)
    plt.close()
