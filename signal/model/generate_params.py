
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pdb

embed_dim = [16,32,64]
num_heads = [1,2]
ff_dim = [16,32,64]
num_layers = [1,2,3]
batch_sizes = [16,32,64]


combos = np.zeros((len(embed_dim)*len(num_heads)*len(ff_dim)*len(num_layers)*len(batch_sizes),5))
i=0
for e in embed_dim:
    for n in num_heads:
        for f in ff_dim:
            for nl in num_layers:
                for s in batch_sizes:
                    combos[i]=[e,n,f,nl,s]
                    i+=1

combo_df = pd.DataFrame()
combo_df['embed_dim']=combos[:,0]
combo_df['num_heads']=combos[:,1]
combo_df['ff_dim']=combos[:,2]
combo_df['num_layers']=combos[:,3]
combo_df['batch_size']=combos[:,4]
combo_df.to_csv('param_combos.csv')
