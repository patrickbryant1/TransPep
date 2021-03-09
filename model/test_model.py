#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter
#Preprocessing and evaluation
from process_data import parse_and_format
#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from categorical_focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint
from multi_head_attention import MultiHeadSelfAttention

from tensorflow.keras.models import model_from_json
import glob


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--json_file', nargs=1, type= str,default=sys.stdin, help = 'path to .json file with keras model to be opened')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test on.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')


#FUNCTIONS
class TransformerBlock(layers.Layer):
    def __init__(self, name, dtype,trainable,embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'ff_dim': ff_dim
        })
        return config

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, name, dtype,trainable,maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': maxlen,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim
        })
        return config

def load_model(json_file, weights):

    json_file = open(json_file, 'r')
    model_json = json_file.read()
    model = model_from_json(model_json,custom_objects = {"TokenAndPositionEmbedding": TokenAndPositionEmbedding, "TransformerBlock": TransformerBlock})
    model.load_weights(weights)
    #print(model.summary())
    return model

def get_data(datadir, test_partition):
    '''Get the validation data
    '''

    train_meta = pd.read_csv(datadir+'train_meta.csv')
    train_seqs = np.load(datadir+'seqs.npy',allow_pickle=True)
    train_annotations = np.load(datadir+'annotations.npy',allow_pickle=True)

    train_CS = train_meta.CS.values
    train_kingdoms = train_meta.Kingdom.values
    train_meta['Type'] = train_meta['Type'].replace({'NO_SP':0,'SP':1,'TAT':2,'LIPO':3})
    train_types = train_meta.Type.values
    #Onehot conversion
    train_kingdoms = np.eye(4)[train_kingdoms]

    #Get data
    #Run through all by taking as input
    test_i = train_meta[train_meta.Partition==test_partition].index
    test_data = []

    test_i = train_meta[train_meta.Partition==test_partition].index
    #valid
    x_test_seqs = train_seqs[test_i]
    x_test_kingdoms = train_kingdoms[test_i]
    x_test = [x_test_seqs,x_test_kingdoms]
    y_test = [train_annotations[test_i],train_types[test_i]]

    return x_test, y_test

def eval_type_cs(pred_annotations,pred_types,true_annotations,true_types,kingdom):
    '''Evaluate the capacity to predict the clevage site
    annotation_conversion = {'S':0,'T':1,'L':2,'I':3,'M':4,'O':5}
    S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide |
    'NO_SP':0,'SP':1,'TAT':2,'LIPO':3
    SP = Sec/SPI
    TAT = Tat/SPI
    LIPO = Sec/SPII

    Reported for CS:
    Recall, TPR = TP/P
    Precision, PPV = TP/(TP+FP)

    Reported for detection
    MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    '''

    Types = {'SP':1,'LIPO':3,'TAT':2}
    Signal_type_annotations = {'SP':0,'TAT':1,'LIPO':2} #S,T,L
    #Save
    fetched_types = []
    MCCs = []
    Recalls = []
    Precisions = []

    if kingdom == 'EUKARYA':
        Types = {'SP':1} #Only one type in Eukarya
    #Go through all types
    for type_name in Types:
        type_enc = Types[type_name]
        P = np.argwhere(true_types==type_enc)[:,0]
        N = np.argwhere(true_types!=type_enc)[:,0]
        #Calc TP and FP
        #Get the pred pos and neg
        pred_P = np.argwhere(pred_types==type_enc)[:,0]
        pred_N = np.argwhere(pred_types!=type_enc)[:,0]
        #TP and TN
        TP = np.intersect1d(P,pred_P).shape[0]
        FP = len(pred_P)-TP
        TN = np.intersect1d(N,pred_N).shape[0]
        FN= len(pred_N)-TN
        #MCC
        MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

        #Get the CS
        type_annotation = Signal_type_annotations[type_name]
        #Get all true positive CSs
        P_annotations = true_annotations[np.intersect1d(P,pred_P)]
        P_CS = []
        for i in range(len(P_annotations)):
            P_CS.append(np.argwhere(P_annotations[i]==type_annotation)[-1,0])

        #Get all pred positive CSs from the true positives (all the other will be wrong)
        P_CS_pred = []
        P_annotations_pred = true_annotations[np.intersect1d(P,pred_P)]
        for i in range(len(P_annotations_pred)):
            P_CS_pred.append(np.argwhere(P_annotations_pred[i]==type_annotation)[-1,0])

        #Get the TP and FP CS
        TP_CS = 0
        FP_CS = 0
        for i in range(len(P_CS)):
            CS_diff = P_CS[i]-P_CS_pred[i]
            if CS_diff<3 and CS_diff>-3:
                TP_CS+=1
            else:
                FP_CS+=1

        #Add the FPs from the wrong detection
        FP_CS += FP
        #Calculate CS precision and recall
        CS_precision = TP_CS/(TP_CS+FP_CS)
        CS_recall = TP_CS/P.shape[0]
        #Save
        fetched_types.append(type_name)
        MCCs.append(MCC)
        Precisions.append(CS_precision)
        Recalls.append(CS_recall)


    return fetched_types, MCCs, Precisions, Recalls

######################MAIN######################
args = parser.parse_args()
json_file = args.json_file[0]
checkpointdir=args.checkpointdir[0]
datadir = args.datadir[0]
test_partition = args.test_partition[0]
outdir = args.outdir[0]

kingdom_conversion = {'ARCHAEA':0,'NEGATIVE':2,'POSITIVE':3,'EUKARYA':1}
#Load and run model
all_pred_annotations = []
all_pred_types = []
all_true_annotations = []
all_true_types = []
all_kingdoms = []

#Get data for each valid partition
for test_partition in np.arange(5):
    #weights
    weights=glob.glob(checkpointdir+str(test_partition)+'/*.hdf5')
    #model
    model = load_model(json_file, weights[0])
    #Get data
    x_test, y_test = get_data(datadir, test_partition)
    pred = model.predict(x_test)
    pred_annotations = np.argmax(pred[0][:,0,:,:],axis=2)
    pred_types = np.argmax(pred[1],axis=1)
    true_annotations = y_test[0]
    true_types = y_test[1]
    kingdoms = np.argmax(x_test[1],axis=1)
    #Save
    all_pred_types.extend([*pred_types])
    all_pred_annotations.extend([*pred_annotations])
    all_true_types.extend([*true_types])
    all_true_annotations.extend([*true_annotations])
    all_kingdoms.extend([*kingdoms])

#Array conversions
all_pred_annotations = np.array(all_pred_annotations)
all_pred_types = np.array(all_pred_types)
all_true_annotations = np.array(all_true_annotations)
all_true_types = np.array(all_true_types)
all_kingdoms = np.array(all_kingdoms)

#Evaluate per kingdom
evaluated_kingdoms = []
all_types = []
all_MCCs = []
all_precisions = []
all_recalls = []
for key in kingdom_conversion:
    kingdom_indices = np.argwhere(all_kingdoms==kingdom_conversion[key])[:,0]
    #Get pred
    kingdom_pred_annotations = all_pred_annotations[kingdom_indices]
    kingdom_pred_types = all_pred_types[kingdom_indices]
    #Get true
    kingdom_true_annotations = all_true_annotations[kingdom_indices]
    kingdom_true_types = all_true_types[kingdom_indices]


#Eval
fetched_types, MCCs, Precisions, Recalls = eval_type_cs(kingdom_pred_annotations,kingdom_pred_types,kingdom_true_annotations,kingdom_true_types,key)

#Save
evaluated_kingdoms.extend([key]*len(fetched_types))
all_types.extend(fetched_types)
all_MCCs.extend(MCCs)
all_precisions.extend(Precisions)
all_recalls.extend(Recalls)

#Create df
eval_df = pd.DataFrame()
eval_df['Kingdom']=evaluated_kingdoms
eval_df['Type']=all_types
eval_df['MCC']=all_MCCs
eval_df['Precision']=all_precisions
eval_df['Recall']=all_recalls
eval_df.to_csv(outdir+'eval_df'+str(test_partition)+'.csv')
print(eval_df)
