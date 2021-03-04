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
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')



###########FUNTIONS###########
def eval_loss(resultsdir,variable_params):
    '''Evaluate the loss
    IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    '''

    #fetch train and valid losses
    test_partition = []
    param_combo = []
    train_losses = []
    valid_losses = []

    train_loss_names = glob.glob(resultsdir+'train_losses*')

    for name in train_loss_names:
        train_loss = np.load(name,allow_pickle=True)
        name = name.split('/')[-1].split('_')
        tp = name[-2]
        test_partition.append(tp)
        pc = name[-1].split('.')[0]
        param_combo.append(pc)
        #get valid data
        try:
            valid_loss = np.load(resultsdir+'valid_losses_'+tp+'_'+pc+'.npy',allow_pickle=True)
            #Save
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
        except:
            continue

    #Create df
    loss_df = pd.DataFrame()
    loss_df['test_partition'] = test_partition
    loss_df['param_combo'] = param_combo
    loss_df = pd.merge(loss_df,variable_params,on='param_combo',how='left')
    #Convert to array
    train_losses = np.array(train_losses)
    valid_losses = np.array(valid_losses)
    #Vis
    best_params = []
    for tp in loss_df['test_partition'].unique():
        sel = loss_df[loss_df.test_partition==tp]

        sel_train_losses = train_losses[sel.index]
        sel_valid_losses = valid_losses[sel.index]

        fig,ax = plt.subplots(figsize=(10/2.54,10/2.54))
        for i in range(len(sel_train_losses)):
            plt.plot(np.arange(train_losses.shape[-1]),np.average(sel_train_losses[i],axis=0),color='tab:blue',alpha=0.1)
            plt.plot(np.arange(train_losses.shape[-1]),np.average(sel_valid_losses[i],axis=0),color='tab:green',alpha=0.1)

        plt.title('Test partition '+tp)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.savefig(resultsdir+'loss_tp_'+tp+'.png',format='png',dpi=300)
        plt.close()
        #Min valid loss vs parameters
        min_valid_losses = np.min(np.average(sel_valid_losses,axis=1),axis=1)
        valid_sel = sel.copy()
        valid_sel['min_valid_loss'] = min_valid_losses
        #Get min combo
        min_combo = valid_sel[valid_sel.min_valid_loss==valid_sel.min_valid_loss.min()]
        best_params.append(min_combo)
        fig,ax = plt.subplots(figsize=(10/2.54,10/2.54))
        sns.pairplot(valid_sel,x_vars=['embed_dim', 'num_heads', 'ff_dim', 'num_layers', 'batch_size'],y_vars='min_valid_loss')
        plt.title('Test partition '+tp)
        plt.tight_layout()
        plt.savefig(resultsdir+'pairplot_tp_'+tp+'.png',format='png',dpi=300)
        plt.close()

    #Save best params
    best_params = pd.concat(best_params)
    best_params.to_csv(resultsdir+'best_params.csv')


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
variable_params=pd.read_csv(args.variable_params[0])            #Unnamed 0 is the index here
variable_params['param_combo']=variable_params['Unnamed: 0']+1 #The sbatch starts from 1, but the df index from 0 --> add 1 to make equal
variable_params['param_combo'] = np.array(variable_params['param_combo'].values,dtype='str')
variable_params = variable_params.drop(columns=['Unnamed: 0'])
#Eval loss
eval_loss(resultsdir,variable_params)
