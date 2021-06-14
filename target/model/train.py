

import argparse
import sys
import os
import numpy as np
import pandas as pd
import time
from collections import Counter

#Keras
import tensorflow as tf

#visualization
from tensorflow.keras.callbacks import TensorBoard

#Custom
from process_data import parse_and_format
from model import create_and_train_model
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A Transformer Neural Network for sorting peptides.''')

parser.add_argument('--datadir', nargs=1, type= str, default=sys.stdin, help = 'Path to data directory.')
parser.add_argument('--variable_params', nargs=1, type= str, default=sys.stdin, help = 'Path to csv with variable params.')
parser.add_argument('--param_combo', nargs=1, type= int, default=sys.stdin, help = 'Parameter combo.')
parser.add_argument('--checkpointdir', nargs=1, type= str, default=sys.stdin, help = 'Path to checpoint directory. Include /in end')
parser.add_argument('--checkpoint', nargs=1, type= int, default=sys.stdin, help = 'If to checkpoint or not: 1= True, 0 = False')
parser.add_argument('--num_epochs', nargs=1, type= int, default=sys.stdin, help = 'Num epochs (int)')
parser.add_argument('--find_lr', nargs=1, type= int, default=sys.stdin, help = 'Find lr (1) or not (0)')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Path to output directory. Include /in end')

#from tensorflow.keras.backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
#sess = tf.Session(config=config)
#set_session(sess)  # set this TensorFlow session as the default session for Keras

#####FUNCTIONS and CLASSES#####

######################MAIN######################
args = parser.parse_args()
datadir = args.datadir[0]
#Get parameters
variable_params=pd.read_csv(args.variable_params[0])
param_combo = args.param_combo[0]
checkpointdir = args.checkpointdir[0]
checkpoint = bool(args.checkpoint[0])
num_epochs = args.num_epochs[0]
find_lr = bool(args.find_lr[0])
outdir = args.outdir[0]

#Params
net_params = variable_params.loc[param_combo-1]
test_partition = int(net_params['test_partition'])
#Fixed params
input_vocab_size = 24  #Amino acids and unknown (X)
target_vocab_size = 9
maxlen = 202  # Only consider the first 200 amino acids + 2 (start and end)


#Get data
try:
    meta = pd.read_csv(datadir+'meta.csv')
    annotations = np.load(datadir+'annotations.npy', allow_pickle=True)
    sequences = np.load(datadir+'sequences.npy', allow_pickle=True)

except:
    data = np.load(datadir+'targetp_data.npz') #'x', 'y_cs', 'y_type', 'len_seq', 'org', 'fold', 'ids'
    meta,  sequences, annotations = parse_and_format(datadir+'targetp.fasta',data)
    #Save
    meta.to_csv(datadir+'meta.csv',index=False)
    np.save(datadir+'annotations.npy',annotations)
    np.save(datadir+'sequences.npy',sequences)




#Get data
#Run through all by taking as input
# Nested cross-validation loop with 5 folds from https://github.com/JJAlmagro/TargetP-2.0/blob/master/train.py
test_i = np.where(meta.Fold ==test_partition)[0]
train_losses = []
valid_losses = []
inner_partitions_interval = np.setdiff1d(np.arange(5),test_partition)
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    print('Validation partition',valid_partition)
    valid_i = np.where(meta.Fold==valid_partition)[0]
    train_i = np.setdiff1d(np.arange(len(meta)),np.concatenate([test_i,valid_i]))

    #Training data
    x_train_inp = sequences[train_i]
    x_train_orgs = np.repeat(np.expand_dims(meta.Org[train_i],axis=1),maxlen,axis=1)
    x_train_tar = annotations[train_i]
    x_train = [x_train_inp, x_train_orgs, x_train_tar]

    #Validation data
    x_valid_inp = sequences[valid_i]
    x_valid_orgs = np.repeat(np.expand_dims(meta.Org[valid_i],axis=1),maxlen,axis=1)
    x_valid_tar = annotations[valid_i]
    x_valid = [x_valid_inp, x_valid_orgs, x_valid_tar]

    #Model
    #Based on: https://keras.io/examples/nlp/text_classification_with_transformer/
    #Variable params
    d_model = int(net_params['embed_dim']) #32  # Embedding size for each token
    num_heads = int(net_params['num_heads']) #1  # Number of attention heads
    dff = int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
    num_layers = int(net_params['num_layers']) #1  # Number of attention heads
    batch_size = int(net_params['batch_size']) #32
    #Create and train model
    create_and_train_model(num_epochs, batch_size, maxlen, input_vocab_size, target_vocab_size, d_model,num_heads, dff,num_layers, x_train, x_valid)
