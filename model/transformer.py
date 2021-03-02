#! /usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import sys
import numpy as np
import pandas as pd
import time
#Preprocessing
from process_data import parse_and_format


#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#visualization
from tensorflow.keras.callbacks import TensorBoard

#from lr_finder import LRFinder


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Transformer Neural Network for analyzing signal peptides.''')

parser.add_argument('--train_data', nargs=1, type= str, default=sys.stdin, help = 'Path to training data in fasta format.')

#parser.add_argument('--params_file', nargs=1, type= str, default=sys.stdin, help = 'Path to file with net parameters')

parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

#####FUNCTIONS and CLASSES#####
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

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


######################MAIN######################
args = parser.parse_args()

try:
    train_meta = pd.read_csv('../data/train_meta.csv')
    train_seqs = np.load('../data/seqs.npy',allow_pickle=True)
    train_annotations = np.load('../data/annotations.npy',allow_pickle=True)
except:
    train_meta, train_seqs, train_annotations = parse_and_format(args.train_data[0])
    #Save
    train_meta.to_csv('../data/train_metata.csv')
    np.save('../data/seqs.npy',train_seqs)
    np.save('../data/annotations.npy',train_annotations)
#params_file = args.params_file[0]
outdir = args.outdir[0]

#Get data
partition=0
valid_i = train_meta[train_meta.Partition==str(partition)].index
train_i = np.setdiff1d(np.arange(len(train_meta)),valid_i)

x_train = train_seqs[train_i]
y_train = train_annotations[train_i]

x_valid = train_seqs[valid_i]
y_valid = train_annotations[valid_i]


#Model
#https://keras.io/examples/nlp/text_classification_with_transformer/


vocab_size = 21  # Only consider the top 20k words
maxlen = 70  # Only consider the first 70 amino acids
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(70, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])


#Summary of model
print(model.summary())

history = model.fit(
    x_train, y_train, batch_size=32, epochs=2, validation_data=(x_valid, y_valid)
)
