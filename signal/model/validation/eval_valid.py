#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import glob


import matplotlib.pyplot as plt
import seaborn as sns

import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Evaluate the validation of the trained models.''')
parser.add_argument('--resultsdir', nargs=1, type= str, default=sys.stdin, help = 'Path to resultsdir.')



###########FUNTIONS###########
def average_validation(valid_df,resultsdir):
    '''Average the MCC, Recall and Precision
    '''
    type_conversion = {'SP':'Sec/SPI','LIPO':'Sec/SPII','TAT':'Tat/SPI'}

    Kingdoms = []
    Types = []
    av_MCC = []
    std_MCC = []
    av_prec = []
    std_prec = []
    av_recall = []
    std_recall = []
    for kingdom in ['ARCHAEA', 'NEGATIVE', 'POSITIVE', 'EUKARYA']:
        sel = valid_df[valid_df.Kingdom==kingdom]
        for type in ['SP', 'LIPO', 'TAT']:
            Kingdoms.append(kingdom)
            Types.append(type_conversion[type])
            sel_type = sel[sel.Type==type]
            av_MCC.append(np.average(sel_type['MCC']))
            std_MCC.append(np.std(sel_type['MCC']))
            av_prec.append(np.average(sel_type['Precision']))
            std_prec.append(np.std(sel_type['Precision']))
            av_recall.append(np.average(sel_type['Recall']))
            std_recall.append(np.std(sel_type['Recall']))

    av_std_df = pd.DataFrame()
    av_std_df['Kingdom']=Kingdoms
    av_std_df['Type']=Types
    av_std_df['MCC average']=av_MCC
    av_std_df['MCC std'] = std_MCC
    av_std_df['Recall average'] = av_recall
    av_std_df['Recall std'] = std_recall
    av_std_df['Precision average'] = av_prec
    av_std_df['Precision std'] = std_prec
    #Save
    av_std_df.to_csv(resultsdir+'valid_results.csv')

    return None

###########MAIN###########
args = parser.parse_args()
resultsdir = args.resultsdir[0]

eval_dfs = glob.glob(resultsdir+'eval_df*.csv')

valid_df = []
for name in eval_dfs:
    df = pd.read_csv(name)
    test_partition = int(name.split('/')[-1][-5])
    df['test_partition']=test_partition
    valid_df.append(df)
valid_df = pd.concat(valid_df)

#Get average scores
average_validation(valid_df,resultsdir)
