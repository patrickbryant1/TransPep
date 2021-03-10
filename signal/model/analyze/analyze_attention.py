#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter

import matplotlib.pyplot as plt
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze attention.''')
parser.add_argument('--attention_dir', nargs=1, type= str, default=sys.stdin, help = '''path to attention.''')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test/bench on.')




#FUNCTIONS



######################MAIN######################
args = parser.parse_args()
attention_dir = args.attention_dir[0]
test_partition = args.test_partition[0]

#Parse
activations1 = []
activations2 = []
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    #Load
    activations1.append(np.load(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations1.npy',allow_pickle=True))
    activations2.append(np.load(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations2.npy',allow_pickle=True))
#Array conversion
activations1 = np.array(activations1)
activations2 = np.array(activations2)
#Average
pdb.set_trace()
