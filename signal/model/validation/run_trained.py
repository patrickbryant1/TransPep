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
from attention_class import MultiHeadSelfAttention #https://apoorvnandan.github.io/2020/05/10/transformer-classifier/


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test on.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')


#FUNCTIONS
#####FUNCTIONS and CLASSES#####
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

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
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

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers,num_iterations):
    '''Create the transformer model
    '''

    seq_input = layers.Input(shape=(maxlen,)) #Input aa sequences
    seq_target = layers.Input(shape=(maxlen,)) #Targets - annotations
    kingdom_input = layers.Input(shape=(4,)) #4 kingdoms, Archaea, Eukarya, Gram +, Gram -

    ##Embeddings
    embedding_layer1 = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    embedding_layer2 = TokenAndPositionEmbedding(maxlen, 6, embed_dim)
    x1 = embedding_layer1(seq_input)
    x2 = embedding_layer2(seq_target)

    #Define the transformer
    transformer_block = TransformerBlock(embed_dim*2, num_heads, ff_dim)
    #Iterate
    for i in range(num_iterations):
        transformer_input = layers.Concatenate()([x1,x2])
        for j in range(num_layers):
            x = transformer_block(transformer_input)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Concatenate()([x,kingdom_input])
        x = layers.Dense(maxlen*6, activation="softmax")(x)
        x_rs = layers.Reshape((maxlen,6))(x)
        x2 = tf.math.argmax(x_rs,axis=-1)
        x2 = embedding_layer2(x2)

    preds = layers.Reshape((maxlen,6),name='annotation')(x)
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

def get_data(datadir, valid_partition):
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
    valid_i = train_meta[train_meta.Partition==valid_partition].index
    #valid
    x_valid_seqs = train_seqs[valid_i]
    x_valid_kingdoms = train_kingdoms[valid_i]
    #Random annotations
    x_valid_target_inp = np.zeros(train_annotations[valid_i].shape)
    x_valid_target_inp[:,:]=np.random.randint(6,size=70)
    x_valid = [x_valid_seqs,x_valid_target_inp,x_valid_kingdoms]
    y_valid = [train_annotations[valid_i],train_types[valid_i]]

    return x_valid_seqs,x_valid_target_inp,x_valid_kingdoms, y_valid

def run_model(model,x_valid_seqs,x_valid_target_inp,x_valid_kingdoms):
    preds = model.predict([x_valid_seqs,x_valid_target_inp,x_valid_kingdoms])

    return preds

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
        if len(P)<1:
            continue
        N = np.argwhere(true_types!=type_enc)[:,0]
        #Calc TP and FP
        #Get the pred pos and neg
        pred_P = np.argwhere(pred_types==type_enc)[:,0]
        pred_N = np.argwhere(pred_types!=type_enc)[:,0]
        #TP and TN
        TP = np.intersect1d(P,pred_P).shape[0]
        if TP<1:
            continue
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
            try:
                CS_precision[d]=TP_CS[d]/(TP_CS[d]+FP_CS[d])
                CS_recall[d] = TP_CS[d]/P.shape[0]
            except:
                pdb.set_trace()


        #Save
        fetched_types.append(type_name)
        MCCs.append(MCC)
        Precisions.append([*CS_precision.values()])
        Recalls.append([*CS_recall.values()])


    return fetched_types, MCCs, Precisions, Recalls


######################MAIN######################
args = parser.parse_args()
variable_params=pd.read_csv(args.variable_params[0])
param_combo=args.param_combo[0]
checkpointdir = args.checkpointdir[0]
datadir = args.datadir[0]
test_partition = args.test_partition[0]
outdir = args.outdir[0]

kingdom_conversion = {'ARCHAEA':0,'NEGATIVE':2,'POSITIVE':3,'EUKARYA':1}
#Load and run model
all_pred_annotations = []
all_true_annotations = []
all_true_types = []
all_kingdoms = []

#Get data for each valid partition
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #weights
    weights=glob.glob(checkpointdir+'vp'+str(valid_partition)+'/*.hdf5')
    if len(weights)<1:
        continue
    #model
    model = load_model(variable_params, param_combo, weights[0])
    #Get data
    x_valid_seqs,x_valid_target_inp,x_valid_kingdoms, y_valid = get_data(datadir, valid_partition)
    #Predict
    preds = run_model(model,x_valid_seqs,x_valid_target_inp,x_valid_kingdoms)
    #Fetch
    pred_annotations = np.argmax(preds,axis=2)
    true_annotations = y_valid[0]
    true_types = y_valid[1]
    kingdoms = np.argmax(x_valid_kingdoms,axis=1)
    #Save
    all_pred_annotations.extend([*pred_annotations])
    all_true_types.extend([*true_types])
    all_true_annotations.extend([*true_annotations])
    all_kingdoms.extend([*kingdoms])



#Array conversions
all_pred_annotations = np.array(all_pred_annotations) #The type will be fetched from the annotations
all_true_annotations = np.array(all_true_annotations)
all_true_types = np.array(all_true_types)
all_kingdoms = np.array(all_kingdoms)
#Get pred types based on pred annotations
all_pred_types = get_pred_types(all_pred_annotations)
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
