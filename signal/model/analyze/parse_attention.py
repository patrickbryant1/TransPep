#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import time
from collections import Counter


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''A program that reads a keras model from a .json and a .h5 file''')
parser.add_argument('--attention_dir', nargs=1, type= str, default=sys.stdin, help = '''path to attention.''')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test/bench on.')




#FUNCTIONS
def parse_attention(attention_file):
    activations1 = []
    activations2 = []
    with open(attention_file, 'r') as file:
        for line in file:
            if line[0:2]=='[[': #Befinning of array
                line = line.strip()
                line = line.strip('[')
                line = line.strip(']')
                line = line.split()
                activations1.append(np.array(line,dtype='float'))
                print('1')
                pdb.set_trace()

            if line[-3:-1]==']]': #End of array
                line = line.strip()
                line = line.strip('[')
                line = line.strip(']]')
                line = line.split()
                activations2.append(np.array(line,dtype='float'))
                print('2')
                pdb.set_trace()

    activations = np.array(activations)
    pdb.set_trace()


######################MAIN######################
args = parser.parse_args()
attention_dir = args.attention_dir[0]
test_partition = args.test_partition[0]

#Parse
for valid_partition in  np.setdiff1d(np.arange(5),test_partition):
    parse_attention(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations.txt')
