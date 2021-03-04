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

def load_model(json_file, weights):

	global model

	json_file = open(json_file, 'r')
	model_json = json_file.read()
	model = model_from_json(model_json)
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
