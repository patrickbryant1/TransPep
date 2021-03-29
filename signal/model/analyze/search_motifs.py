#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import time
from collections import Counter
import pdb

sys.path.insert(0, "../")
from process_data import parse_and_format

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Search the benchmark sequences for motifs.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = '''Path to data.''')

#####################FUNCTIONS#####################
def convert_bench_seqs(bench_seqs):
    '''Convert the bench seqs back to character format
    '''

    AMINO_ACIDS = { 0:'A',1:'R',2:'N',3:'D',4:'C',5:'E',
                    6:'Q',7:'G',8:'H',9:'I',10:'L',11:'K',
                    12:'M',13:'F',14:'P',15:'S',16:'T',17:'W',
                    18:'Y',19:'V',20:'X'
                  }

    conv_seqs = []
    for seq in bench_seqs:
        conv_seq = ''
        for aa in seq:
            conv_seq+=AMINO_ACIDS[aa]

        conv_seqs.append(conv_seq)

    return conv_seqs

def search_motifs():
    '''
    '''


######################MAIN######################
plt.rcParams.update({'font.size': 7})
args = parser.parse_args()
datadir = args.datadir[0]
bench_meta = pd.read_csv(datadir+'bench_meta.csv')
bench_seqs = np.load(datadir+'bench_seqs.npy',allow_pickle=True)
bench_annotations = np.load(datadir+'bench_annotations.npy',allow_pickle=True)

#Convert to character format
conv_seqs = convert_bench_seqs(bench_seqs)

pdb.set_trace()
for type in data.Type.unique():
    type_data = data[data.Type==type]
    type_Seqs = Seqs[type_data.index]
