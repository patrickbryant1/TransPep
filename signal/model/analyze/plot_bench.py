#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt


import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Obtain attention activations.''')
parser.add_argument('--benchcsv', nargs=1, type= str, default=sys.stdin, help = 'Path to bench csv.')
parser.add_argument('--outdir', nargs=1, type= str, default=sys.stdin, help = 'Output directory.')


#FUNCTIONS


######################MAIN######################
args = parser.parse_args()
benchcsv = pd.read_csv(args.benchcsv[0])
outdir = args.outdir[0]
