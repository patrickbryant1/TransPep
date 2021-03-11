#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.insert(0, "../")
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
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--bench_set', nargs=1, type= str, default=sys.stdin, help = 'Path to benchmark dataset.')
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


def get_activations(seq_inp):
    '''Get token and position embedding and attention activations
    '''
    #token and position
    get_token_position_emb = keras.backend.function([model.layers[0].input],[model.layers[1].output])
    token_position_emb = get_token_position_emb(seq_inp)[0]

    #attention
    # get_attention = keras.backend.function([model.layers[0].input],[model.layers[3].output])
    # get_attention = get_attention(seq_inp)[0]

    return token_position_emb

def process_bench_set(bench_set,datadir):
    bench_meta, bench_seqs, bench_annotations = parse_and_format(bench_set)
    #Save
    bench_meta.to_csv(datadir+'bench_meta.csv')
    np.save(datadir+'bench_seqs.npy',bench_seqs)
    np.save(datadir+'bench_annotations.npy',bench_annotations)

    return None

def get_data(datadir, test_partition):
    '''Get the test data
    '''

    #Get the test data
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

    #test
    x_test_seqs = train_seqs[test_i]
    x_test_kingdoms = train_kingdoms[test_i]
    x_test = [x_test_seqs,x_test_kingdoms]
    y_test = [train_annotations[test_i],train_types[test_i]]

    #Get the bench data
    bench_meta = pd.read_csv(datadir+'bench_meta.csv')
    bench_seqs = np.load(datadir+'bench_seqs.npy',allow_pickle=True)
    bench_annotations = np.load(datadir+'bench_annotations.npy',allow_pickle=True)


    bench_kingdoms = bench_meta.Kingdom.values
    bench_meta['Type'] = bench_meta['Type'].replace({'NO_SP':0,'SP':1,'TAT':2,'LIPO':3})
    bench_types = bench_meta.Type.values
    #Onehot conversion
    bench_kingdoms = np.eye(4)[bench_kingdoms]

    #Get data
    #Run through all by taking as input
    bench_i = bench_meta[bench_meta.Partition==test_partition].index

    #bench
    x_bench_seqs = bench_seqs[bench_i]
    x_bench_kingdoms = bench_kingdoms[bench_i]
    x_bench = [x_bench_seqs,x_bench_kingdoms]
    y_bench = [bench_annotations[bench_i],bench_types[bench_i]]

    return x_test, y_test, x_bench, y_bench

def eval_type_cs(pred_annotations,pred_types,true_annotations,true_types,kingdom):
    '''Evaluate the capacity to predict the clevage site
    annotation_conversion = {'S':0,'T':1,'L':2,'I':3,'M':4,'O':5}
    annotation [S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]
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
        P_annotations_pred = pred_annotations[np.intersect1d(P,pred_P)]
        for i in range(len(P_annotations_pred)):
            try:
                P_CS_pred.append(np.argwhere(P_annotations_pred[i]==type_annotation)[-1,0])
            except:
                P_CS_pred.append(0)


        #Get the TP and FP CS
        TP_CS = {0:0,1:0,2:0,3:0} #exact CS, +/-1 error, +/-2 error, +/-3 error
        FP_CS = {0:0,1:0,2:0,3:0}
        for i in range(len(P_CS)):
            CS_diff = P_CS[i]-P_CS_pred[i]
            for d in range(0,4):
                if CS_diff<=d and CS_diff>=-d:
                    TP_CS[d]+=1
                else:
                    FP_CS[d]+=1

        #Add the FPs from the wrong detection
        for d in range(0,4):
            FP_CS[d] += FP

        #Calculate CS precision and recall
        CS_precision = {}
        CS_recall = {}
        for d in range(0,4):
            CS_precision[d]=TP_CS[d]/(TP_CS[d]+FP_CS[d])
            CS_recall[d] = TP_CS[d]/P.shape[0]


        #Save
        fetched_types.append(type_name)
        MCCs.append(MCC)
        Precisions.append([*CS_precision.values()])
        Recalls.append([*CS_recall.values()])


    return fetched_types, MCCs, Precisions, Recalls


def eval_preds(all_pred_annotations, all_pred_types,all_true_annotations,all_true_types,all_kingdoms,mode, outdir):
    '''Evaluate the predictions per kingdom and signal peptide
    '''
    kingdom_conversion = {'ARCHAEA':0,'NEGATIVE':2,'POSITIVE':3,'EUKARYA':1}

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
    eval_df['Recall [0,1,2,3]']=all_recalls
    eval_df['Precision [0,1,2,3]']=all_precisions
    #Rename
    eval_df = eval_df.replace({'Type': {'SP':'Sec/SPI','LIPO':'Sec/SPII','TAT':'Tat/SPI'}})
    eval_df.to_csv(outdir+'nested_'+mode+'_eval_df.csv')
    print(eval_df)


######################MAIN######################
args = parser.parse_args()
checkpointdir=args.checkpointdir[0]
datadir = args.datadir[0]
bench_set = args.bench_set[0]
outdir = args.outdir[0]

#Preprocess the bench set
#process_bench_set(bench_set,datadir)

#Load and run model
#test
test_all_pred_annotations = []
test_all_pred_types = []
test_all_true_annotations = []
test_all_true_types = []
test_all_kingdoms = []
#bench
bench_all_pred_annotations = []
bench_all_pred_types = []
bench_all_true_annotations = []
bench_all_true_types = []
bench_all_kingdoms = []
bench_all_seqs = []
bench_all_token_pos_emb = []
#Get data for each test partition
for test_partition in np.arange(5):
    #json file with model description
    json_file = checkpointdir+'TP'+str(test_partition)+'/model.json'
    #test
    test_pred_annotations = []
    test_pred_types = []
    #bench
    bench_pred_annotations = []
    bench_pred_types = []
    bench_token_pos_emb = []


    #Get data
    x_test, y_test, x_bench, y_bench = get_data(datadir, test_partition)

    for valid_partition in  np.setdiff1d(np.arange(5),test_partition):
        #weights
        weights=glob.glob(checkpointdir+'TP'+str(test_partition)+'/vp'+str(valid_partition)+'/*.hdf5')
        #model
        model = load_model(json_file, weights[0])

        #Predict
        test_pred = model.predict(x_test)
        bench_pred = model.predict(x_bench)

        #Get token_position_emb
        bench_token_pos_emb.append(get_activations(x_bench[0]))

        #Save
        #test
        test_pred_annotations.append(test_pred[0][:,0,:,:])
        test_pred_types.append(test_pred[1])
        #bench
        bench_pred_annotations.append(bench_pred[0][:,0,:,:])
        bench_pred_types.append(bench_pred[1])


    #Join all nested preds
    #TEST
    #Annotations
    test_pred_annotations = np.array(test_pred_annotations)
    test_pred_annotations = np.average(test_pred_annotations,axis=0)
    test_pred_annotations = np.argmax(test_pred_annotations,axis=2)
    #Types
    test_pred_types = np.array(test_pred_types)
    test_pred_types = np.average(test_pred_types,axis=0)
    test_pred_types = np.argmax(test_pred_types,axis=1)
    test_true_annotations = y_test[0]
    test_true_types = y_test[1]
    test_kingdoms = np.argmax(x_test[1],axis=1)
    #Save
    test_all_pred_types.extend([*test_pred_types])
    test_all_pred_annotations.extend([*test_pred_annotations])
    test_all_true_types.extend([*test_true_types])
    test_all_true_annotations.extend([*test_true_annotations])
    test_all_kingdoms.extend([*test_kingdoms])

    #BENCH
    #Annotations
    bench_pred_annotations = np.array(bench_pred_annotations)
    bench_pred_annotations = np.average(bench_pred_annotations,axis=0)
    bench_pred_annotations = np.argmax(bench_pred_annotations,axis=2)
    #Types
    bench_pred_types = np.array(bench_pred_types)
    bench_pred_types = np.average(bench_pred_types,axis=0)
    bench_pred_types = np.argmax(bench_pred_types,axis=1)
    bench_true_annotations = y_bench[0]
    bench_true_types = y_bench[1]
    bench_kingdoms = np.argmax(x_bench[1],axis=1)
    #TokenAndPositionEmbedding
    bench_token_pos_emb = np.array(bench_token_pos_emb)
    bench_token_pos_emb = np.average(bench_token_pos_emb,axis=0)
    #Save
    bench_all_pred_types.extend([*bench_pred_types])
    np.save(checkpointdir+'TP'+str(test_partition)+'/bench_pred_types.npy',bench_pred_types)
    bench_all_pred_annotations.extend([*bench_pred_annotations])
    bench_all_true_types.extend([*bench_true_types])
    np.save(checkpointdir+'TP'+str(test_partition)+'/bench_true_types.npy',bench_true_types)
    bench_all_true_annotations.extend([*bench_true_annotations])
    bench_all_kingdoms.extend([*bench_kingdoms])

    #TokenAndPositionEmbedding
    np.save(checkpointdir+'TP'+str(test_partition)+'/bench_token_position_emb.npy',bench_token_pos_emb)
    #Save the input sequences
    np.save(checkpointdir+'TP'+str(test_partition)+'/bench_seqs',x_bench[0])

#Array conversions
#test
test_all_pred_annotations = np.array(test_all_pred_annotations)
test_all_pred_types = np.array(test_all_pred_types)
test_all_true_annotations = np.array(test_all_true_annotations)
test_all_true_types = np.array(test_all_true_types)
test_all_kingdoms = np.array(test_all_kingdoms)
#bench
bench_all_pred_annotations = np.array(bench_all_pred_annotations)
bench_all_pred_types = np.array(bench_all_pred_types)
bench_all_true_annotations = np.array(bench_all_true_annotations)
bench_all_true_types = np.array(bench_all_true_types)
bench_all_kingdoms = np.array(bench_all_kingdoms)


#Eval
eval_preds(test_all_pred_annotations, test_all_pred_types,test_all_true_annotations,test_all_true_types,test_all_kingdoms,'test', outdir)
eval_preds(bench_all_pred_annotations, bench_all_pred_types,bench_all_true_annotations,bench_all_true_types,bench_all_kingdoms,'bench', outdir)
pdb.set_trace()
