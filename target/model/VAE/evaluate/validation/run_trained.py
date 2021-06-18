#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.insert(0, "../../")
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
from model import create_model
#Umap
import umap

import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')


def load_model(net_params, vocab_size, maxlen, weights):

    #Variable params
    embed_dim = int(net_params['embed_dim']) #32  # Embedding size for each token
    num_heads = int(net_params['num_heads']) #1  # Number of attention heads
    ff_dim = int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
    num_layers = int(net_params['num_layers']) #1  # Number of attention heads
    batch_size = int(net_params['batch_size']) #32
    #Create model
    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers, False)
    model.load_weights(weights)

    return model

def get_data(datadir, valid_partition):
    '''Get the validation data
    '''

    meta = pd.read_csv(datadir+'meta.csv')
    CS = meta.CS.values
    Types = meta.Type.values
    sequences = np.load(datadir+'sequences.npy', allow_pickle=True)
    #Valid data
    valid_i = np.where(meta.Fold==valid_partition)[0]

    #Validation data
    x_valid = sequences[valid_i]

    #Get the true types and CS
    true_types = Types[valid_i]
    true_CS = CS[valid_i]

    return x_valid, true_types, true_CS

def get_attention_and_encodings(model,x_valid):
    '''Obtain the output of the attention layers
    '''
    # Names
    names = [weight.name for layer in model.layers for weight in layer.weights]
    #Self-attention
    #get_enc_self_attention = keras.backend.function(model.layers[0].input, )
    #enc_attention = get_enc_layer_output(x_valid)

    #Endodings
    get_encodings = keras.backend.function(model.layers[0].input, model.get_layer('encoder').output)
    encodings_z = get_encodings(x_valid)
    return encodings_z #enc_attention


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
vocab_size = 22  #Amino acids and unknown (X)
maxlen = 200  # Only consider the first 70 amino acids

all_true_types = []
all_true_CS = []
all_encodings_z = []
#Load and run model for each valid partition
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #weights
    weights=glob.glob(checkpointdir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/*.hdf5')
    #model
    model = load_model(net_params, vocab_size, maxlen, weights[0])
    #Get data
    x_valid, true_types, true_CS = get_data(datadir, valid_partition)
    #Get attention and encodings
    encodings_z = get_attention_and_encodings(model,x_valid)
    #Save
    #True
    all_true_types.extend([*true_types])
    all_true_CS.extend([*true_CS])
    all_encodings_z.extend([*encodings_z])


#Array conversions
all_encodings_z = np.array(all_encodings_z)

#Umap
mapper = umap.UMAP().fit(all_encodings_z)
pdb.set_trace()
