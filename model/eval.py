#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import glob
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Evaluate the trained models.''')

parser.add_argument('--resultsdir', nargs=1, type= str, default=sys.stdin, help = 'Path to resultsdir.')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')


###########FUNTIONS###########
def eval_loss():
    '''Parse and format the data:
    >Uniprot_AC|Kingdom|Type|Partition No
    amino-acid sequence
    annotation [S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]

    >P36001|EUKARYA|NO_SP|1
    MDDISGRQTLPRINRLLEHVGNPQDSLSILHIAGTNGKETVSKFLTSILQHPGQQRQRVLIGRYTTSSLL
    IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    '''

    #fetch train losses
    train_losses = glob.glob(resultsdir+'train_losses*')
    
    for name in train_losses:


def eval_cs(preds,true):
    '''Evaluate the capacity to predict the clevage site
    annotation_conversion = {'S':0,'T':1,'L':2,'I':3,'M':4,'O':5}
    S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide |
    '''

    true_s = []
    pred_s = []
    true_t = []
    pred_t = []
    true_l = []
    pred_l = []

    missing_s = 0
    total_s = 0
    missing_t = 0
    total_t = 0
    missing_l = 0
    total_l = 0
    for i in range(len(preds)):
        if 0 in true[i]:
            total_s+=1
            if 0 in preds[i]:
                true_s.append(np.argwhere(true[i]==0)[-1,0])
                pred_s.append(np.argwhere(preds[i]==0)[-1,0])
            else:
                missing_s+=1
        if 1 in true[i]:
            total_t+=1
            if 1 in preds[i]:
                true_t.append(np.argwhere(true[i]==1)[-1,0])
                pred_t.append(np.argwhere(preds[i]==1)[-1,0])
            else:
                missing_t+=1
        if 2 in true[i]:
            total_l+=1
            if 2 in preds[i]:
                true_l.append(np.argwhere(true[i]==2)[-1,0])
                pred_l.append(np.argwhere(preds[i]==2)[-1,0])
            else:
                missing_l +=1


    sns.distplot(np.array(pred_s)-np.array(true_s), label='Sec/SPI signal peptide: '+str(np.round(missing_s/total_s,2)))
    sns.distplot(np.array(pred_t)-np.array(true_t),label='Tat/SPI signal peptide: '+str(np.round(missing_t/total_t,2)))
    sns.distplot(np.array(pred_l)-np.array(true_l),label='Sec/SPII signal peptide: '+str(np.round(missing_l/total_l,2)))
    plt.legend()
    plt.show()
    pdb.set_trace()


###########MAIN###########
args = parser.parse_args()
resultsdir = args.resultsdir[0]
datadir = args.datadir[0]
outdir = args.outdir[0]
