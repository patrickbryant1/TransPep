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


import glob
from categorical_focal_loss import SparseCategoricalFocalLoss
from attention_class import MultiHeadSelfAttention

import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combos', nargs=1, type= str, default=sys.stdin, help = 'Parameter combos.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--bench_set', nargs=1, type= str, default=sys.stdin, help = 'Path to benchmark dataset.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')


#FUNCTIONS
#FUNCTIONS
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(EncoderBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, in_q,in_k,in_v, training): #Inputs is a list with [q,k,v]
        attn_output,attn_weights = self.att(in_q,in_k,in_v) #The weights are needed for downstream analysis
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(in_q + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_weights

class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(DecoderBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim,num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, in_q,in_k,in_v, training): #Inputs is a list with [q,k,v]
        #Self-attention
        attn_output1,attn_weights1 = self.att(in_q,in_q,in_q) #The weights are needed for downstream analysis
        attn_output1 = self.dropout1(attn_output1, training=training)
        out1 = self.layernorm1(in_v + attn_output1)
        #Encoder-decoder attention
        attn_output2,attn_weights2 = self.att(out1,in_k,in_v) #The weights are needed for downstream analysis
        attn_output2 = self.dropout1(attn_output2, training=training)
        out2 = self.layernorm1(attn_output2 + attn_output1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output), attn_weights2

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers,num_iterations):
    '''Create the transformer model
    '''

    seq_input = layers.Input(shape=(maxlen,)) #Input aa sequences
    seq_target = layers.Input(shape=(maxlen,)) #Targets - annotations
    kingdom_input = layers.Input(shape=(maxlen,4)) #4 kingdoms, Archaea, Eukarya, Gram +, Gram -

    ##Embeddings
    embedding_layer1 = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    embedding_layer2 = TokenAndPositionEmbedding(maxlen, 6, embed_dim+4) #Need to add 4 so that x1 and x2 match
    x1 = embedding_layer1(seq_input)
    #Add kingdom input
    x1 = layers.Concatenate()([x1,kingdom_input])
    x2 = embedding_layer2(seq_target)

    #Define the transformer
    encoder = EncoderBlock(embed_dim+4, num_heads, ff_dim)
    decoder = DecoderBlock(embed_dim+4, num_heads, ff_dim)
    #Iterate
    for i in range(num_iterations):
        #Encode
        for j in range(num_layers):
            x1, enc_attn_weights = encoder(x1,x1,x1) #q,k,v
        #Decoder
        for k in range(num_layers):
            x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder

        x2 = layers.Dense(6, activation="softmax")(x2) #Annotate
        x_rs = layers.Reshape((maxlen,6))(x2)
        x2 = tf.math.argmax(x_rs,axis=-1) #Needed for iterative training
        x2 = embedding_layer2(x2)

    x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder
    preds = layers.Dense(6, activation="softmax")(x2) #Annotate
    #preds = layers.Reshape((maxlen,6),name='annotation')(x2)
    #pred_type = layers.Dense(4, activation="softmax",name='type')(x) #Type of protein
    #pred_cs = layers.Dense(1, activation="elu", name='pred_cs')(x)


    model = keras.Model(inputs=[seq_input,seq_target,kingdom_input], outputs=preds)

    return model


def load_model(variable_params, param_combo, weights):

    #Params
    net_params = variable_params.loc[param_combo-1]
    #Fixed params
    vocab_size = 21  # Only consider the top 20k words
    maxlen = 70  # Only consider the first 70 amino acids
    #Variable params
    embed_dim = int(net_params['embed_dim']) #32  # Embedding size for each token
    num_heads = int(net_params['num_heads']) #1  # Number of attention heads
    ff_dim = int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
    num_layers = int(net_params['num_layers']) #1  # Number of attention heads
    batch_size = int(net_params['batch_size']) #32
    num_iterations = int(net_params['num_iterations'])
    #Create model
    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers,num_iterations)
    model.load_weights(weights)
    #print(model.summary())

    return model


def process_bench_set(bench_set,datadir):
    bench_meta, bench_seqs, bench_annotations = parse_and_format(bench_set)
    #Save
    bench_meta.to_csv(datadir+'bench_meta.csv')
    np.save(datadir+'bench_seqs.npy',bench_seqs)
    np.save(datadir+'bench_annotations.npy',bench_annotations)

    return None

def get_data(datadir, test_partition, maxlen):
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
    x_test_kingdoms = np.repeat(np.expand_dims(x_test_kingdoms,axis=1),70,axis=1)
    #Random annotations
    x_test_target_inp =  np.random.randint(6,size=(len(test_i),maxlen))
    x_test = [x_test_seqs,x_test_target_inp,x_test_kingdoms]
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
    x_bench_kingdoms = np.repeat(np.expand_dims(x_bench_kingdoms,axis=1),70,axis=1)
    #Random annotations
    x_bench_target_inp =  np.random.randint(6,size=(len(bench_i),maxlen))
    x_bench = [x_bench_seqs, x_bench_target_inp, x_bench_kingdoms]
    y_bench = [bench_annotations[bench_i],bench_types[bench_i]]

    return x_test, y_test, x_bench, y_bench

def get_pred_types(pred_annotations):
    '''Get the predicted types based on the annotations
    '''

    annotation_type_conversion = {0:1,1:2,2:3} #S(0)=SP(1), T(1)=TAT(2),L(2)=LIPO(3) - all other 0 (No SP)
    pred_types = []
    for i in range(len(pred_annotations)):
        if (0 in pred_annotations[i]) or (1 in pred_annotations[i]) or (2 in pred_annotations[i]):
            counts = Counter(pred_annotations[i])
            keys = [*counts.keys()]

            key_count=0 #Count the occurance of each annotation - take the max for the type
            key_type = 0
            for key in annotation_type_conversion: #Got through all keys
                if key not in keys:
                    continue
                else:
                    if counts[key]>key_count:
                        key_count=counts[key]
                        key_type=annotation_type_conversion[key]

            #Save
            pred_types.append(key_type)

        else:
            pred_types.append(0)

    return np.array(pred_types)

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
#Get parameters
variable_params=pd.read_csv(args.variable_params[0])
param_combos=args.param_combos[0]
param_combos = param_combos.split(',')
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

    param_combo = int(param_combos[test_partition])

    #test
    test_pred_annotations = []
    test_pred_types = []
    #bench
    bench_pred_annotations = []
    bench_pred_types = []
    bench_token_pos_emb = []



    #Get data
    x_test, y_test, x_bench, y_bench = get_data(datadir, test_partition,70)

    for valid_partition in  np.setdiff1d(np.arange(5),test_partition):
        #weights
        weights=glob.glob(checkpointdir+'TP'+str(test_partition)+'/vp'+str(valid_partition)+'/*.hdf5')
        #model
        model = load_model(variable_params, param_combo, weights[0])
        #Predict
        test_pred = model.predict(x_test)
        bench_pred = model.predict(x_bench)

        #Save
        #test
        test_pred_annotations.append(test_pred)
        #bench
        bench_pred_annotations.append(bench_pred)

    #Join all nested preds
    #TEST
    #Annotations
    test_pred_annotations = np.array(test_pred_annotations)
    test_pred_annotations = np.average(test_pred_annotations,axis=0)
    test_pred_annotations = np.argmax(test_pred_annotations,axis=2)
    #Types
    test_pred_types = get_pred_types(test_pred_annotations)

    #True
    test_true_annotations = y_test[0]
    test_true_types = y_test[1]
    test_kingdoms = np.argmax(x_test[2][:,0],axis=1)
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
    bench_pred_types = get_pred_types(bench_pred_annotations)
    #True
    bench_true_annotations = y_bench[0]
    bench_true_types = y_bench[1]
    bench_kingdoms = np.argmax(x_bench[2][:,0],axis=1)

    #Save
    bench_all_pred_types.extend([*bench_pred_types])
    bench_all_pred_annotations.extend([*bench_pred_annotations])
    bench_all_true_types.extend([*bench_true_types])
    bench_all_true_annotations.extend([*bench_true_annotations])
    bench_all_kingdoms.extend([*bench_kingdoms])


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
