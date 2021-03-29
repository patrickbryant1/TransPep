#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from collections import Counter
import re
import pdb

sys.path.insert(0, "../")
from process_data import parse_and_format

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Search the benchmark sequences for motifs.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = '''Path to data.''')
parser.add_argument('--kingdom', nargs=1, type= int, default=sys.stdin, help = '''Kingdom.''')
parser.add_argument('--type', nargs=1, type= str, default=sys.stdin, help = '''SP type.''')
parser.add_argument('--nmotif', nargs=1, type= str, default=sys.stdin, help = '''N-terminal motif.''')
parser.add_argument('--cmotif', nargs=1, type= str, default=sys.stdin, help = '''C-terminal motif.''')

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

    return np.array(conv_seqs)

def search_motifs(seqs,pattern):
    '''
    pattern='(R|K)(D|T|M|R)(K|S|M|Q)'+'.'*24+'(I|F|V|T|S|A)'+'.'+'(S|A)'
    '''

    matches=0
    for seq in seqs:
        result = re.findall(pattern, seq)
        if result:
            matches+=len(result)

    return matches



######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
type = args.type[0]
kingdom = args.kingdom[0]
nmotif = args.nmotif[0].split(',')
cmotif = args.cmotif[0]
bench_meta = pd.read_csv(datadir+'bench_meta.csv')
bench_seqs = np.load(datadir+'bench_seqs.npy',allow_pickle=True)
bench_annotations = np.load(datadir+'bench_annotations.npy',allow_pickle=True)

#Convert to character format
conv_seqs = convert_bench_seqs(bench_seqs)
#Get data
non_type_data = bench_meta.loc[(bench_meta.Type!=type) & (bench_meta.Kingdom==kingdom)]
non_type_seqs = conv_seqs[non_type_data.index]
type_data = bench_meta.loc[(bench_meta.Type==type) & (bench_meta.Kingdom==kingdom)]
type_seqs = conv_seqs[type_data.index]
#search seqs
pattern=''.join(nmotif)+'.'*20+cmotif
print(pattern)
nt_matches = search_motifs(non_type_seqs,pattern)
t_matches = search_motifs(type_seqs,pattern)
print('N & C-terminal')
print('NO_SP',nt_matches,',',type,t_matches)
