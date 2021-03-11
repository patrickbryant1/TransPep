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
from transformer_classes import MultiHeadAttention, Encoder, Decoder

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
class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):

        enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        #final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return dec_output, attention_weights

###MASKING###
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)

  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)

  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

  return enc_padding_mask, combined_mask, dec_padding_mask

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers):
    '''Create the transformer model
    '''

    seq_input = layers.Input(shape=(maxlen,)) #Input aa sequences
    seq_target = layers.Input(shape=(None,)) #Targets - annotations
    kingdom_input = layers.Input(shape=(4,)) #4 kingdoms, Archaea, Eukarya, Gram +, Gram -

    #Define the transformer
    transformer = Transformer(num_layers, embed_dim, num_heads, ff_dim, 21, 7,maxlen,maxlen)
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(seq_input,seq_target)
    x, attention_weights = transformer(seq_input,seq_target,
                    True,
                    enc_padding_mask, combined_mask, dec_padding_mask)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Concatenate()([x,kingdom_input])
    preds = layers.Dense(maxlen*7, activation="softmax")(x)
    preds = layers.Reshape((-1,maxlen,7),name='annotation')(preds)
    pred_type = layers.Dense(4, activation="softmax",name='type')(x) #Type of protein
    #pred_cs = layers.Dense(1, activation="elu", name='pred_cs')(x)


    model = keras.Model(inputs=[seq_input,seq_target,kingdom_input], outputs=[preds,pred_type])
    #Optimizer
    opt = keras.optimizers.Adam(learning_rate=0.001,amsgrad=True)
    #Compile
    model.compile(optimizer = opt, loss= [SparseCategoricalFocalLoss(gamma=2),SparseCategoricalFocalLoss(gamma=2)], metrics=["accuracy"])

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

    #Create model
    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers)

    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers)
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
    #The annotation 6 will be added to the train annotations as a start token (the annotations range from 0-5)
    x_valid_target_inp=np.zeros((len(valid_i),1))
    x_valid_target_inp[:,0]=6
    y_valid = [train_annotations[valid_i],train_types[valid_i]]

    return x_valid_seqs,x_valid_target_inp,x_valid_kingdoms, y_valid

def run_model(model,x_valid_seqs,x_valid_target_inp,x_valid_kingdoms):

    #Run predicions for all positions
    for i in range(70):
        print(i)
        #Predict
        preds = model.predict([x_valid_seqs,x_valid_target_inp,x_valid_kingdoms])
        #Update x_valid_target_inp
        x_valid_target_inp = np.zeros((x_valid_target_inp.shape[0],i+2))
        x_valid_target_inp[:,0]=6
        #Here beam search can be implemented
        x_valid_target_inp[:,1:]=np.argmax(preds[0][:,0,:i+1,:],axis=2)
        pdb.set_trace()
    return None


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
        P_annotations_pred = pred_annotations[np.intersect1d(P,pred_P)]
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
variable_params=pd.read_csv(args.variable_params[0])
param_combo=args.param_combo[0]
checkpointdir = args.checkpointdir[0]
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
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #weights
    weights=glob.glob(checkpointdir+'vp'+str(valid_partition)+'/*.hdf5')
    #model
    model = load_model(variable_params, param_combo, weights[0])
    #Get data
    x_valid_seqs,x_valid_target_inp,x_valid_kingdoms, y_valid = get_data(datadir, valid_partition)
    #Predict
    run_model(model,x_valid_seqs,x_valid_target_inp,x_valid_kingdoms)
    #Fetch
    pred_annotations = np.argmax(pred[0][:,0,:,:],axis=2)
    pred_types = np.argmax(pred[1],axis=1)
    true_annotations = y_valid[0]
    true_types = y_valid[1]
    kingdoms = np.argmax(x_valid[1],axis=1)
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
