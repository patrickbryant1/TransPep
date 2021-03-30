#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import time
from collections import Counter
import pdb

#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Analyze uniprot reference genomes.''')
parser.add_argument('--meta', nargs=1, type= str, default=sys.stdin, help = '''Path to preotein meta file in tsv.''')

#####################FUNCTIONS#####################



######################MAIN######################
args = parser.parse_args()
meta = pd.read_csv(args.meta[0],sep='\t')

pdb.set_trace()

#['Entry', 'Entry name', 'Status', 'Protein names', 'Gene names',
#'Organism', 'Length', 'Organism ID', 'Signal peptide']

#Fraction SP
non_sp = meta['Signal peptide'].dropna().shape
sp = len(meta)-non_sp
print('Num SP',sp,'Fraction SP',np.round(sp/len(meta),3))
#meta.Organism.unique().shape: (1841,)
