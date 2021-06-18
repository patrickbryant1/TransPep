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
#Umap
import umap

import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Runs saved models and obtain intermediate layer outputs''')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
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

    return x_valid, true_CS, true_types, true_orgs, true_IDs

def get_attention_and_encodings(model,x_valid):
    '''Obtain the output of the attention layers
    '''
    # Names
    #encoder_layers = [layer.name for layer in model.get_layer('encoder').layers]
    #Self-attention
    get_enc_self_attention = keras.backend.function(model.layers[0].input, model.get_layer('encoder').get_layer('encoder_block').output)
    _, enc_attention = get_enc_self_attention(x_valid)

    #Endodings
    get_encodings = keras.backend.function(model.layers[0].input, model.get_layer('encoder').output)
    encodings_z = get_encodings(x_valid)
    return encodings_z, enc_attention


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

all_true_CS = []
all_true_types = []
all_true_orgs = []
all_true_IDs = []
all_encodings_z = []
all_enc_attention = []
all_seqs = []
#Load and run model for each valid partition
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #weights
    weights=glob.glob(checkpointdir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/*.hdf5')
    if len(weights)<1:
        continue
    #model
    model = load_model(net_params, vocab_size, maxlen, weights[0])
    #Get data
    x_valid, true_CS, true_types, true_orgs, true_IDs = get_data(datadir, valid_partition)
    #Get attention and encodings
    encodings_z, enc_attention = get_attention_and_encodings(model,x_valid)
    #Save
    all_true_CS.extend([*true_CS])
    all_true_types.extend([*true_types])
    all_true_orgs.extend([*true_orgs])
    all_true_IDs.extend([*true_IDs])
    all_encodings_z.extend([*encodings_z])
    all_enc_attention.extend([*np.max(enc_attention,axis=1)])
    all_seqs.extend([*x_valid])


#Array conversions
all_encodings_z = np.array(all_encodings_z)
all_seqs = np.array(all_seqs)

#Umap
print('Mapping UMAP for seqs...')
us_seq = umap.UMAP().fit_transform(all_seqs)
print('Mapping UMAP for encodings...')
us = umap.UMAP().fit_transform(all_encodings_z)
#Save
np.save(outdir+'umap_seqs'+str(test_partition)+'.npy',us_seq)
np.save(outdir+'umap'+str(test_partition)+'.npy',us)
np.save(outdir+'enc_attention'+str(test_partition)+'.npy',np.array(all_enc_attention))
np.save(outdir+'CS'+str(test_partition)+'.npy',np.array(all_true_CS))
np.save(outdir+'types'+str(test_partition)+'.npy',np.array(all_true_types))
np.save(outdir+'orgs'+str(test_partition)+'.npy',np.array(all_true_orgs))
np.save(outdir+'IDs'+str(test_partition)+'.npy',np.array(all_true_IDs))
