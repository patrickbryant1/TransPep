

import argparse
import sys
import os
import numpy as np
import pandas as pd
import time
from collections import Counter

#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from categorical_focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint


#visualization
from tensorflow.keras.callbacks import TensorBoard

from multi_head_attention import MultiHeadSelfAttention
#from lr_finder import LRFinder


import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Transformer Neural Network for sorting peptides.''')

parser.add_argument('--train_data', nargs=1, type= str, default=sys.stdin, help = 'Path to training data in fasta format.')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test on.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = 'Path to checpoint directory. Include /in end')
parser.add_argument('--save_model', nargs=1, type= int, default=sys.stdin, help = 'If to save model or not: 1= True, 0 = False')
parser.add_argument('--checkpoint', nargs=1, type= int, default=sys.stdin, help = 'If to checkpoint or not: 1= True, 0 = False')
parser.add_argument('--num_epochs', nargs=1, type= int, default=sys.stdin, help = 'Num epochs (int)')
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'maxlen': maxlen,
            'vocab_size': vocab_size,
            'embed_dim': embed_dim
        })
        return config

######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
#Get parameters
#variable_params=pd.read_csv(args.variable_params[0])
test_partition = args.test_partition[0]
checkpointdir = args.checkpointdir[0]
save_model = bool(args.save_model[0])
checkpoint = bool(args.checkpoint[0])
num_epochs = args.num_epochs[0]
outdir = args.outdir[0]

#Load data
data = np.load(datadir+'targetp_data.npz')
folds = data['fold']
#Get data
#Run through all by taking as input
# Nested cross-validation loop with 5 folds from https://github.com/JJAlmagro/TargetP-2.0/blob/master/train.py
partitions_interval = np.arange(5)
for test_partition in partitions_interval:
    inner_partitions_interval = partitions_interval[partitions_interval != test_partition]
    # Inner cross-validation
    for val_partition in inner_partitions_interval:
        # Create directory to store the model
        try:
            os.mkdir('%spartition_%i_%i' % (outdir, test_partition, val_partition))
        except:
            print('%spartition_%i_%i' % (outdir, test_partition, val_partition),'exists')
        model_file = "%spartition_%i_%i/model" % (outdir, test_partition, val_partition)

        # Define train and validation splits
        train_partition = inner_partitions_interval[inner_partitions_interval != val_partition]
        train_set = np.in1d(folds.ravel(), train_partition).reshape(folds.shape)
        val_set = np.where(folds == val_partition)

        # Load training data
        x_train = data['x'][train_set] #x_train.shape (7796, 200, 20) - 200 longx20 onehot enc aa
        y_train = data['y_cs'][train_set] #y_train.shape (7796, 200) - 200 long, CS marked with a 1, rest 0


        len_train = data['len_seq'][train_set]
        org_train = data['org'][train_set]
        #5 classes of transit peptides
        #0=no targeting peptide, 1=sp: signal peptide, 2=mt:mitochondrial transit peptide,
        #3=ch:chloroplast transit peptide, 4=th:thylakoidal lumen composite transit peptide
        tp_train = data['y_type'][train_set]

        # Load validation data
        x_val = data['x'][val_set]
        y_val = data['y_cs'][val_set]
        len_val = data['len_seq'][val_set]
        org_val = data['org'][val_set]
        tp_val = data['y_type'][val_set]

        pdb.set_trace()
