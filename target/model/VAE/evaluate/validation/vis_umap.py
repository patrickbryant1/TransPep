#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.insert(0, "../../")
import numpy as np
import pandas as pd
import time

import glob
import matplotlib.pyplot as plt
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory containing UMAP of embeddings and raw sequences.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')


def vis_umap(ar):
    '''Visualize UMAP
    '''
    #5 classes of transit peptides
    #0 = pad
    #1=no targeting peptide/Inside cell, 2=sp: signal peptide, 3=mt:mitochondrial transit peptide,
    #4=ch:chloroplast transit peptide, 5=th:thylakoidal lumen composite transit peptide
    #6=Outside of cell - only valid for SPs - not for the peptides going into mt or ch/th
    #Colors
    colors = {1:'b',2:'r',3:'k',4:'orange',5:'magenta'}
    type_colors = [colors[i] for i in all_true_types]

    plt.scatter(us[:,0], us[:,1], c=type_colors,s=2,alpha=0.5)
######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
outdir = args.outdir[0]

#Get seq and emb projections
for i in range(5):
    seqs = np.load(outdir+'umap_seqs'+str(i)+'.npy',allow_pickle=True)
    emb = np.load(outdir+'umap'+str(i)+'.npy',allow_pickle=True)
