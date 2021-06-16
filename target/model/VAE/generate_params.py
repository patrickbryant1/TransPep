
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pdb

encode_dim = [16,32]
num_heads = [1,4,8]
batch_sizes = [32,64]
test_partitions = [0,1,2,3,4]

combos = np.zeros((len(encode_dim)*len(num_heads)*len(batch_sizes)*len(test_partitions),4))
i=0
for e in encode_dim:
    for n in num_heads:
        for s in batch_sizes:
            for tp in test_partitions:
                combos[i]=[e,n,s,tp]
                i+=1

combo_df = pd.DataFrame()
combo_df['encode_dim']=combos[:,0]
combo_df['num_heads']=combos[:,1]
combo_df['batch_size']=combos[:,2]
combo_df['test_partition']=combos[:,3]
combo_df.to_csv('param_combos.csv')
