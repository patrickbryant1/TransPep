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
import umap
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory containing UMAP of embeddings and raw sequences.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')


def vis_umap(us, us_seq, all_types, all_orgs):
    '''Visualize UMAP
    '''
    #5 classes of transit peptides
    #0 = pad
    #1=no targeting peptide/Inside cell, 2=sp: signal peptide, 3=mt:mitochondrial transit peptide,
    #4=ch:chloroplast transit peptide, 5=th:thylakoidal lumen composite transit peptide
    #6=Outside of cell - only valid for SPs - not for the peptides going into mt or ch/th
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
    #Plot

    #Types enc
    fig,ax = plt.subplots(figsize=(12/2.54,12/2.54))
    for type in df.type.unique():
        sel = df[df.type==type]
        plt.scatter(sel['u1'],sel['u2'],s=2,color=type_colors[type],label=type,alpha=0.5)
    plt.legend()
    plt.title('Embeddings')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('u1')
    plt.ylabel('u2')
    plt.tight_layout()
    plt.savefig(outdir+'type_umap.png',dpi=300)

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
    plt.savefig(outdir+'type_umap_seq.png',dpi=300)

    #Org enc
    fig,ax = plt.subplots(figsize=(12/2.54,12/2.54))
    for org in df.org.unique():
        sel = df[df.org==org]
        plt.scatter(sel['u1'],sel['u2'],s=2,color=org_colors[org],label=org,alpha=0.5)
    plt.legend()
    plt.title('Embeddings')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('u1')
    plt.ylabel('u2')
    plt.tight_layout()
    plt.savefig(outdir+'org_umap.png',dpi=300)

    #Org enc
    fig,ax = plt.subplots(figsize=(12/2.54,12/2.54))
    for org in df.org.unique():
        sel = df[df.org==org]
        plt.scatter(sel['u1_seq'],sel['u2_seq'],s=2,color=org_colors[org],label=org,alpha=0.5)
    plt.legend()
    plt.title('Sequences')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('u1')
    plt.ylabel('u2')
    plt.tight_layout()
    plt.savefig(outdir+'org_umap_seq.png',dpi=300)

    pdb.set_trace()
######################MAIN######################
args = parser.parse_args()
#Set font size
matplotlib.rcParams.update({'font.size': 7})
datadir = args.datadir[0]
outdir = args.outdir[0]

#Get seq and emb projections
for i in [0]:
    all_encodings_z = []
    all_types = []
    all_orgs = []
    all_seqs = []
    for j in np.setdiff1d(range(5),i):
        all_encodings_z.extend([*np.load(outdir+'enc_z'+str(i)+'_'+str(j)+'.npy',allow_pickle=True)])
        all_types.extend([*np.load(outdir+'types'+str(i)+'_'+str(j)+'.npy',allow_pickle=True)])
        all_orgs.extend([*np.load(outdir+'orgs'+str(i)+'_'+str(j)+'.npy',allow_pickle=True)])
        all_seqs.extend([*np.load(outdir+'seqs'+str(i)+'_'+str(j)+'.npy',allow_pickle=True)])
    # #Umap

    try:
        us =np.load(datadir+'umap'+str(i)+'.npy',allow_pickle=True)
        us_seq =np.load(datadir+'umap_seq'+str(i)+'.npy',allow_pickle=True)
    except:
        print('Mapping UMAP for encodings...')
        us = umap.UMAP().fit_transform(all_encodings_z)
        print('Mapping UMAP for seqs...')
        us_seq = umap.UMAP().fit_transform(all_seqs)
        #save
        np.save(datadir+'umap'+str(i)+'.npy',us)
        np.save(datadir+'umap_seq'+str(i)+'.npy',us_seq)
    #Visualize
    vis_umap(us,us_seq, all_types, all_orgs)
