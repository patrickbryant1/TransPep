#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter
#Preprocessing and evaluation
from process_data import parse_and_format, eval_cs

#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from categorical_focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint
from multi_head_attention import MultiHeadSelfAttention

from tensorflow.keras.models import model_from_json


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--json_file', nargs=1, type= str,default=sys.stdin, help = 'path to .json file with keras model to be opened')
parser.add_argument('--weights', nargs=1, type= str, default=sys.stdin, help = '''path to .h5 file containing weights for net.''')
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

    global model

    json_file = open(json_file, 'r')
    model_json = json_file.read()
    pdb.set_trace()
    model = model_from_json(model_json,custom_objects = {"TokenAndPositionEmbedding": TokenAndPositionEmbedding(), "TransformerBlock": TransformerBlock()})

    # Retrieve the config
    config = model.get_config()
    # At loading time, register the custom objects with a `custom_object_scope`:
    with keras.utils.custom_object_scope(custom_objects):
        new_model = keras.Model.from_config(config)

    model.load_weights(weights)
    model._make_predict_function()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

######################MAIN######################
args = parser.parse_args()
json_file = args.json_file[0]
weights = args.weights[0]
datadir = args.datadir[0]
test_partition = args.test_partition[0]
outdir = args.outdir[0]

#Load and run model
model = load_model(json_file, weights)
pred = model.predict(X_valid)

pdb.set_trace()
