

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
from model import create_model
from lr_finder import LRFinder
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''An attention based VAE Neural Network for encoding peptides.''')

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
vocab_size = 22  #Amino acids, unknown (X) and padding
maxlen = 200  # Only consider the first 200 amino acids


#Get data
try:
    meta = pd.read_csv(datadir+'meta.csv')
    sequences = np.load(datadir+'sequences.npy', allow_pickle=True)

except:
    data = np.load(datadir+'targetp_data.npz') #'x', 'y_cs', 'y_type', 'len_seq', 'org', 'fold', 'ids'
    meta,  sequences = parse_and_format(datadir+'targetp.fasta',data)
    #Save
    meta.to_csv(datadir+'meta.csv',index=False)
    np.save(datadir+'sequences.npy',sequences)

#Get data
#Run through all by taking as input
# Nested cross-validation loop with 5 folds from https://github.com/JJAlmagro/TargetP-2.0/blob/master/train.py
test_i = np.where(meta.Fold ==test_partition)[0]
train_losses = []
valid_losses = []
train_acc = []
valid_acc = []
inner_partitions_interval = np.setdiff1d(np.arange(5),test_partition)
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    print('Validation partition',valid_partition)
    valid_i = np.where(meta.Fold==valid_partition)[0]
    train_i = np.setdiff1d(np.arange(len(meta)),np.concatenate([test_i,valid_i]))

    #Training data
    x_train = sequences[train_i]
    #Validation data
    x_valid = sequences[valid_i]
    #Model
    #Based on: https://keras.io/examples/nlp/text_classification_with_transformer/
    #Variable params
    embed_dim = int(net_params['embed_dim']) #32  # Embedding size for each token
    num_heads = int(net_params['num_heads']) #1  # Number of attention heads
    ff_dim = int(net_params['ff_dim']) #32  # Hidden layer size in feed forward network inside transformer
    num_layers = int(net_params['num_layers']) #1  # Number of attention heads
    batch_size = int(net_params['batch_size']) #32
    #Create and train model
    model = create_model(maxlen, vocab_size, embed_dim,num_heads, ff_dim,num_layers, find_lr)

    if find_lr == True:
        lr_finder = LRFinder(model)
        lr_finder.find(x_train, x_train, start_lr=0.00001, end_lr=1, batch_size=batch_size, epochs=1)
        losses = lr_finder.losses
        lrs = lr_finder.lrs
        l_l = np.asarray([lrs, losses])
        np.savetxt(outdir+'lrs_losses'+str(param_combo)+'.txt', l_l)
        num_epochs = 0
        break

    else:
        #Checkpoint
        if checkpoint == True:
            #Make dir
            try:
                os.mkdir(checkpointdir+'TP'+str(test_partition)+'/VP'+str(valid_partition))
            except:
                print('Checkpoint directory exists...')

            checkpoint_path=checkpointdir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+"/weights_"+str(param_combo)+"_{epoch:02d}.hdf5"
            checkpointer = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
            save_weights_only=True, mode='auto', save_freq='epoch')

            #Callbacks
            callbacks=[checkpointer]
        else:
            callbacks = []

        history = model.fit(x=x_train,y=x_train,
            epochs=num_epochs,
            validation_data=(x_valid,x_valid),
            callbacks=[]
            )

        #Save loss
        train_losses.append(history.history['loss'])
        valid_losses.append(history.history['val_loss'])
        train_acc.append(history.history['accuracy'])
        valid_acc.append(history.history['val_accuracy'])

if find_lr != True and checkpoint != True:
    #Save array of losses
    outid = str(test_partition)+'_'+str(param_combo)
    np.save(outdir+'train_losses_'+outid+'.npy',np.array(train_losses))
    np.save(outdir+'valid_losses_'+outid+'.npy',np.array(valid_losses))
    print('Done')
