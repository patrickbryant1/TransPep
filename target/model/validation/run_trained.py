#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.insert(0, "../")
import numpy as np
import pandas as pd
import time
from collections import Counter
#Preprocessing
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
        out1 = self.layernorm1(in_q + attn_output1)
        #Encoder-decoder attention
        attn_output2,attn_weights2 = self.att(out1,in_k,in_v) #The weights are needed for downstream analysis
        attn_output2 = self.dropout1(attn_output2, training=training)
        out2 = self.layernorm1(attn_output2 + out1)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out2 + ffn_output), attn_weights2

def create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers,num_iterations):
    '''Create the transformer model
    '''

    seq_input = layers.Input(shape=(maxlen,)) #Input aa sequences
    seq_target = layers.Input(shape=(maxlen,)) #Targets - annotations
    org_input = layers.Input(shape=(maxlen,2)) #2 Organisms plant/not

    ##Embeddings
    embedding_layer1 = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    embedding_layer2 = TokenAndPositionEmbedding(maxlen, 5, embed_dim+2) #5 annotation classes. Need to add 2 so that x1 and x2 match - the organsims
    x1 = embedding_layer1(seq_input)
    #Add kingdom input
    x1 = layers.Concatenate()([x1,org_input])
    x2 = embedding_layer2(seq_target)

    #Define the transformer
    encoder = EncoderBlock(embed_dim+2, num_heads, ff_dim)
    decoder = DecoderBlock(embed_dim+2, num_heads, ff_dim)
    #Iterate
    for i in range(num_iterations):
        #Encode
        for j in range(num_layers):
            x1, enc_attn_weights = encoder(x1,x1,x1) #q,k,v
        #Decoder
        for k in range(num_layers):
            x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder

        x2 = layers.Dense(5, activation="softmax")(x2) #Annotate
        x_rs = layers.Reshape((maxlen,5))(x2)
        x2 = tf.math.argmax(x_rs,axis=-1) #Needed for iterative training
        x2 = embedding_layer2(x2)

    x2, enc_dec_attn_weights = decoder(x2,x1,x1) #q,k,v - the k and v from the encoder goes into he decoder
    preds = layers.Dense(5, activation="softmax")(x2) #Annotate


    model = keras.Model(inputs=[seq_input,seq_target,org_input], outputs=preds)
    #Optimizer
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    decay_rate=0.96,
    staircase=True)

    opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule,amsgrad=True)

    #Compile
    model.compile(optimizer = opt, loss= SparseCategoricalFocalLoss(gamma=2), metrics=["accuracy"])

    return model


def load_model(net_params, vocab_size, maxlen, weights):

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

def get_data(datadir, valid_partition, maxlen, vocab_size):
    '''Get the validation data
    '''

    meta = pd.read_csv(datadir+'meta.csv')
    annotations = np.load(datadir+'annotations.npy', allow_pickle=True)
    sequences = np.load(datadir+'sequences.npy', allow_pickle=True)
    #Valid data
    valid_i = np.where(meta.Fold==valid_partition)[0]

    #Validation data
    x_valid_seqs = sequences[valid_i]
    x_valid_orgs = np.repeat(np.expand_dims(np.eye(2)[meta.Org[valid_i]],axis=1),maxlen,axis=1)
    #Random annotations are added as input
    x_valid_target_inp = np.random.randint(5,size=(len(valid_i),maxlen))
    y_valid = annotations[valid_i] #,train_types[train_i]]

    #Get the true types and CS
    true_types = meta.Type.values[valid_i]
    true_CS = meta.CS.values[valid_i]

    return x_valid_seqs,x_valid_orgs, x_valid_target_inp, y_valid, true_types, true_CS

def run_model(model,x_valid):
    preds = model.predict(x_valid)

    return preds

def get_pred_types(pred_annotations):
    '''Get the predicted types based on the annotations
    '''

    #5 classes of transit peptides
    #0=no targeting peptide, 1=sp: signal peptide, 2=mt:mitochondrial transit peptide,
    #3=ch:chloroplast transit peptide, 4=th:thylakoidal lumen composite transit peptide
    type_conversion = {1:'SP', 2:'MT', 3:'CH', 4:'TH'}

    pred_types = []
    for i in range(len(pred_annotations)):
        if (1 in pred_annotations[i]) or (2 in pred_annotations[i]) or (3 in pred_annotations[i]) or (4 in pred_annotations[i]):
            counts = Counter(pred_annotations[i])
            keys = [*counts.keys()]

            key_count=0 #Count the occurance of each annotation - take the max for the type
            key_type = 0
            for key in type_conversion: #Got through all keys
                if key not in keys:
                    continue
                else:
                    if counts[key]>key_count:
                        key_count=counts[key]
                        key_type=key

            #Save
            pred_types.append(key_type)

        else:
            pred_types.append(0)

    return np.array(pred_types)


def eval_type_cs(pred_annotations, pred_types, true_annotations, true_types, true_CS):
    '''
    5 classes of transit peptides
    0=no targeting peptide, 1=sp: signal peptide, 2=mt:mitochondrial transit peptide,
    3=ch:chloroplast transit peptide, 4=th:thylakoidal lumen composite transit peptide

    Reported for CS:
    Recall, TPR = TP/P

    Reported for detection (correct type)
    Recall, TPR = TP/P
    Precision, PPV = TP/(TP+FP)
    F1 = 2/(1/Recall + 1/Precision)
    MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    '''

    type_conversion = {0:'No target', 1:'SP', 2:'MT', 3:'CH', 4:'TH'}
    #Save
    CS_recalls = {0:[],5:[]}
    type_recalls = []
    type_precisions = []
    F1s = []
    MCCs = []


    #Go through all types
    for type in type_conversion:

        #Type metrics
        P = np.argwhere(true_types==type)[:,0]
        N = np.argwhere(true_types!=type)[:,0]
        #Calc TP and FP
        #Get the pred pos and neg
        pred_P = np.argwhere(pred_types==type)[:,0]
        pred_N = np.argwhere(pred_types!=type)[:,0]
        pdb.set_trace()
        #TP and TN
        TP = np.intersect1d(P,pred_P).shape[0]
        if TP<1:
            continue
        FP = len(pred_P)-TP
        TN = np.intersect1d(N,pred_N).shape[0]
        FN= len(pred_N)-TN

        #Precision and recall
        type_recalls.append(TP/P.shape[0])
        type_precisions.append(TP/(TP+FP))
        #F1
        F1s.append(2/(1/type_recalls[-1]+1/type_precisions[-1]))
        #MCC
        MCCs.append((TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))

        #CS metrics
        #Get all true positive CSs
        P_CS  = true_CS[np.intersect1d(P,pred_P)]

        #Get all pred positive CSs from the true positives (all the other will be wrong)
        P_CS_pred = []
        P_annotations_pred = pred_annotations[np.intersect1d(P,pred_P)]
        for i in range(len(P_annotations_pred)):
            P_CS_pred.append(np.argwhere(P_annotations_pred[i]==type)[-1,0])

        #Get the TP and FP CS
        TP_CS = {0:0,5:0} #exact CS, +/-1 error, +/-2 error, +/-3 error
        FP_CS = {0:0,5:0}
        for i in range(len(P_CS)):
            CS_diff = P_CS[i]-P_CS_pred[i]
            for d in [0,5]:
                if CS_diff<=d and CS_diff>=-d:
                    TP_CS[d]+=1
                else:
                    FP_CS[d]+=1

        #Add the FPs from the wrong detection
        for d in [0,5]:
            FP_CS[d] += FP

        #Calculate CS precision and recall
        for d in [0,5]:
            CS_recalls[d] = TP_CS[d]/P.shape[0]

    pdb.set_trace()
    #Create df
    eval_df = pd.DataFrame()
    eval_df['Type']=[*type_conversion.values()]
    eval_df['CS_recall_0'] = CS_recalls[0]
    eval_df['CS_recall_5'] = CS_recalls[5]
    eval_df['Type_recall']=type_recalls
    eval_df['Type_precision']=type_precisions
    eval_df['Type_F1']=F1s
    eval_df['Type_MCC']=MCCs

    return eval_df


######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
variable_params=pd.read_csv(args.variable_params[0])
param_combo=args.param_combo[0]
checkpointdir = args.checkpointdir[0]
outdir = args.outdir[0]

#Params
net_params = variable_params.loc[param_combo-1]
test_partition = int(net_params['test_partition'])

#Fixed params
vocab_size = 21  #Amino acids and unknown (X)
maxlen = 200  # Only consider the first 70 amino acids

#Load and run model for each valid partition
all_pred_annotations = []
all_true_annotations = []
all_true_types = []
all_true_CS = []

for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #weights
    weights=glob.glob(checkpointdir+'VP'+str(valid_partition)+'/*.hdf5')
    #model
    model = load_model(net_params, vocab_size, maxlen, weights[0])
    #Get data
    x_valid_seqs,x_valid_orgs, x_valid_target_inp, y_valid, true_types, true_CS = get_data(datadir, valid_partition, maxlen, vocab_size)
    #Predict
    x_valid = [x_valid_seqs,x_valid_target_inp,x_valid_orgs] #inp seq, target annoation, organism
    preds = run_model(model,x_valid)

    #Pred and true annotations
    y_pred = np.argmax(preds,axis=2)

    #Save
    #Pred
    all_pred_annotations.extend([*y_pred])
    #True
    all_true_annotations.extend([*y_valid])
    all_true_types.extend([*true_types])
    all_true_CS.extend([*true_CS])



#Array conversions
all_pred_annotations = np.array(all_pred_annotations)
#The type will be fetched from the annotations
all_pred_types = get_pred_types(all_pred_annotations)
all_true_annotations = np.array(all_true_annotations)
all_true_types = np.array(all_true_types)
all_true_CS = np.array(all_true_CS)

#Eval
eval_df = eval_type_cs(all_pred_annotations, all_pred_types, all_true_annotations, all_true_types, all_true_CS)

eval_df.to_csv(outdir+'eval_df'+str(test_partition)+'.csv')
print(eval_df)
