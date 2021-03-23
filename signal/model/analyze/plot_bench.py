#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Obtain attention activations.''')
parser.add_argument('--benchcsv', nargs=1, type= str, default=sys.stdin, help = 'Path to bench csv.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Output directory.')


##########FUNCTIONS###########
def plot_scores(type_CSV,type,outdir):
    '''Plot the distributions
    '''
    if type == 'Sec/SPI':
        height=6/2.54
        aspect=9/4.5
    else:
         height=6/(2.54)
         aspect=4.5/4.5

    for score in ['MCC','Recall','Precision']:

        sns.color_palette("Set2")
        sns.set_style("whitegrid")
        fig,ax = plt.subplots(figsize=(180/2.54,6/2.54))
        g = sns.catplot(x='Method', y=score, hue='Kingdom', kind="bar",
        data=type_CSV, palette="Paired",legend=False, height=height, aspect=aspect)
        g.set_xticklabels(rotation=45)
        plt.ylim([0,1])
        plt.title(type+' '+score)
        plt.tight_layout()
        plt.savefig(outdir+'_'.join(type.split('/'))+'_'+score+'.png',dpi=300)





######################MAIN######################
args = parser.parse_args()
benchcsv = pd.read_csv(args.benchcsv[0])
outdir = args.outdir[0]
matplotlib.rcParams.update({'font.size': 7})

#Fix data
benchcsv = benchcsv.replace('n.d.',0)
for score in ['MCC','Recall','Precision']:
    benchcsv[score] = np.array(benchcsv[score],dtype='int')/1000
for type in benchcsv.Type.unique():
    type_CSV = benchcsv[benchcsv.Type==type]
    plot_scores(type_CSV,type,outdir)
