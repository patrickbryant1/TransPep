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
meta = pd.read_csv(args.meta[0])
