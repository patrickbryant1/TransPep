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
from tensorflow.keras import regularizers,optimizers
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling1D,add,Lambda,Dense, Dropout, Activation, Conv1D, BatchNormalization, Flatten, Subtract
from tensorflow.keras.losses import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
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

#FUNCTIONS




######################MAIN######################
args = parser.parse_args()

try:
    train_data = pd.read_csv('../data/train_data.csv')
    train_annotations = np.load('../data/annotations.npy',allow_pickle=True)
except:
    train_data, train_annotations = parse_and_format(args.train_data[0])
    #Save
    train_data.to_csv('../data/train_data.csv')
    np.save('../data/annotations.npy',train_annotations)
#params_file = args.params_file[0]
outdir = args.outdir[0]

valid_i = train_data[train_data.Partition==0].index
train_i = np.setdiff1d(np.arange(len(train_data)),valid_i)

X_train = train_data.loc[train_i,'Sequence'].values
y_train = train_annotations[train_i]

X_valid = train_data.loc[valid_i,'Sequence'].values
y_valid = train_annotations[valid_i]
pdb.set_trace()

#Summary of model
#print(model.summary())
