
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pdb

embed_dim = [14,30] #2 will be added --> 16 and 32
num_heads = [1,4,8]
ff_dim = [16,32]
num_layers = [1,2,4]
batch_sizes = [32,64]
num_iterations = [1,2,4]
test_partitions = [0,1,2,3,4]

combos = np.zeros((len(embed_dim)*len(num_heads)*len(ff_dim)*len(num_layers)*len(batch_sizes)*len(num_iterations)*len(test_partitions),7))
i=0
for e in embed_dim:
    for n in num_heads:
        for f in ff_dim:
            for nl in num_layers:
                for s in batch_sizes:
                    for ni in num_iterations:
                        for tp in test_partitions:
                            combos[i]=[e,n,f,nl,s,ni,tp]
                            i+=1

combo_df = pd.DataFrame()
combo_df['embed_dim']=combos[:,0]
combo_df['num_heads']=combos[:,1]
combo_df['ff_dim']=combos[:,2]
combo_df['num_layers']=combos[:,3]
combo_df['batch_size']=combos[:,4]
combo_df['num_iterations']=combos[:,5]
combo_df['test_partition']=combos[:,6]
combo_df.to_csv('param_combos.csv')
