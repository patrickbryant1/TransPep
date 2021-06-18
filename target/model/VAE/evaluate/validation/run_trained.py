#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
sys.path.insert(0, "../../")
import numpy as np
import pandas as pd
import time
from collections import Counter

#Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import glob
from model import create_model
import matplotlib.pyplot as plt


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Runs saved models and obtain intermediate layer outputs''')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--valid_partition', nargs=1, type= int, default=sys.stdin, help = 'Valid partition.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = '''path checkpoints with .h5 files containing weights for net.''')
parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = '''path to output dir.''')


def load_model(net_params, vocab_size, maxlen, weights):

    #Variable params
    embed_dim = int(net_params['embed_dim']) # Embedding size for each token
    num_heads = int(net_params['num_heads'])  # Number of attention heads
    ff_dim = int(net_params['ff_dim'])  # Hidden layer size in feed forward network inside transformer and the final embedding size
    num_layers = int(net_params['num_layers']) # Number of attention heads
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
    Orgs = meta.Org.values
    IDs = meta.ID.values
    sequences = np.load(datadir+'sequences.npy', allow_pickle=True)
    #Valid data
    valid_i = np.where(meta.Fold==valid_partition)[0]

    #Validation data
    x_valid = sequences[valid_i]

    #Get the true types and CS

    true_CS = CS[valid_i]
    true_types = Types[valid_i]
    true_orgs = Orgs[valid_i]
    true_IDs = IDs[valid_i]
    seqs = sequences[valid_i]

    return x_valid, true_CS, true_types, true_orgs, true_IDs, seqs

def get_attention_and_encodings(model,x_valid):
    '''Obtain the output of the attention layers
    '''
    # Names
    encoder_layers = [layer.name for layer in model.get_layer('encoder').layers]
    for name in encoder_layers:
        if 'encoder_block' in name:
            break

    #Self-attention
    get_enc_self_attention = keras.backend.function(model.layers[0].input, model.get_layer('encoder').get_layer(name).output)
    _, enc_attention = get_enc_self_attention(x_valid)

    #Endodings
    get_encodings = keras.backend.function(model.layers[0].input, model.get_layer('encoder').get_layer('z').output)
    encodings_z = get_encodings(x_valid)
    return encodings_z, enc_attention


######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
variable_params=pd.read_csv(args.variable_params[0])
param_combo=args.param_combo[0]
valid_partition=args.valid_partition[0]
checkpointdir = args.checkpointdir[0]
outdir = args.outdir[0]

#Params
net_params = variable_params.loc[param_combo-1]
test_partition = int(net_params['test_partition'])

#Fixed params
vocab_size = 22  #Amino acids and unknown (X)
maxlen = 200  # Only consider the first 70 amino acids

#Load and run model for each valid partition
#weights
weights=glob.glob(checkpointdir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/*.hdf5')
#model
model = load_model(net_params, vocab_size, maxlen, weights[0])
#Get data
x_valid, true_CS, true_types, true_orgs, true_IDs, seqs = get_data(datadir, valid_partition)
#Get attention and encodings
encodings_z, enc_attention = get_attention_and_encodings(model,x_valid)
#Save
np.save(outdir+'enc_z'+str(test_partition)+'_'+str(valid_partition)+'.npy',np.array(encodings_z))
np.save(outdir+'enc_attention'+str(test_partition)+'_'+str(valid_partition)+'.npy',np.array(enc_attention))
np.save(outdir+'CS'+str(test_partition)+'_'+str(valid_partition)+'.npy',np.array(true_CS))
np.save(outdir+'types'+str(test_partition)+'_'+str(valid_partition)+'.npy',np.array(true_types))
np.save(outdir+'orgs'+str(test_partition)+'_'+str(valid_partition)+'.npy',np.array(true_orgs))
np.save(outdir+'IDs'+str(test_partition)+'_'+str(valid_partition)+'.npy',np.array(true_IDs))
np.save(outdir+'seqs'+str(test_partition)+'_'+str(valid_partition)+'.npy',seqs)
