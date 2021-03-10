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
parser = argparse.ArgumentParser(description = '''Parse attention.''')
parser.add_argument('--attention_dir', nargs=1, type= str, default=sys.stdin, help = '''path to attention.''')
parser.add_argument('--test_partition', nargs=1, type= int, default=sys.stdin, help = 'Which CV fold to test/bench on.')




#FUNCTIONS
def parse_attention(attention_file):
    activations1 = []
    activations2 = []
    ln=0
    new_ar = []
    fetch=False
    with open(attention_file, 'r') as file:
        for line in file:
            ln+=1
            print(ln)
            line = line.strip()
            if '[[[' in line:
                fetch = True

            if fetch==False:
                continue

            ##Get end
            if ']]]]' in line:
                line = line.strip('[')
                line = line.strip(']]]]')
                line = line.split()
                new_ar.append(np.array(line,dtype='float'))
                activations2.append(np.array(new_ar))
                new_ar = [] #reset
                continue


            line = line.strip('[')
            line = line.strip(']')



            if ']]' in line: #Two ars in line
                line = line.split()
                split_ar = []
                for item in line:
                    if ']]]' in item:
                        item = item[:-3]
                        split_ar.append(item)
                        new_ar.append(np.array(split_ar,dtype='float'))
                        continue

                    if '[[[[' in item:
                        split_ar = []
                        item = item[4:]

                    split_ar.append(item)

                activations1.append(np.array(new_ar))
                #Create a new array
                new_ar = []
                new_ar.append(np.array(split_ar,dtype='float'))

            else:
                line = line.split()
                new_ar.append(np.array(line,dtype='float'))






    activations1 = np.array(activations1)
    activations2 = np.array(activations2)

    return activations1, activations2


######################MAIN######################
args = parser.parse_args()
attention_dir = args.attention_dir[0]
test_partition = args.test_partition[0]

#Parse
for valid_partition in np.setdiff1d(np.arange(5),test_partition):
    activations1, activations2 = parse_attention(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations.txt')
    print(valid_partition)
    #Save
    np.save(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations1.npy',activations1)
    np.save(attention_dir+'TP'+str(test_partition)+'/VP'+str(valid_partition)+'/activations2.npy',activations2)
