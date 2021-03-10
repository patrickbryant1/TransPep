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
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test/bench on.')
parser.add_argument('--valid_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to get the valid model from.')



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
    '''Get the test data
    '''

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

    return x_bench, y_bench


######################MAIN######################
args = parser.parse_args()
checkpointdir=args.checkpointdir[0]
datadir = args.datadir[0]
test_partition = args.test_partition[0]
valid_partition = args.valid_partition[0]


#Load and run model

#json file with model description
json_file = checkpointdir+'TP'+str(test_partition)+'/model.json'
#test

#Get data
x_bench, y_bench = get_data(datadir, test_partition)
#Get activations - printed to stderr in multi_head_attention function
#weights
weights=glob.glob(checkpointdir+'TP'+str(test_partition)+'/vp'+str(valid_partition)+'/*.hdf5')
#model
model = load_model(json_file, weights[0])
#Predict
for i in range(len(x_bench[0])):
    bench_pred = model.predict([np.array([x_bench[0][i]]),np.array([x_bench[1][i]])])
