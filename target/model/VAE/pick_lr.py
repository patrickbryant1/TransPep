
#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import glob
import pdb

lrates = glob.glob('../../results/VAE/lrate/*.txt')
plt.rcParams.update({'font.size': 7})
combo = []
fig,ax = plt.subplots(figsize=(12/2.54,9/2.54))
for name in lrates:
    combo.append(int(name.split('/')[-1].split('.')[0].split('losses')[-1]))
    lrs_losses= np.loadtxt(name)
    plt.plot(lrs_losses[0,:],lrs_losses[1,:],alpha=0.1,color='tab:blue')

plt.xscale('log')
plt.ylim([0,10])
plt.title('Learning rate optimization')
plt.xlabel('learning rate')
plt.ylabel('loss')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_color('mediumseagreen')
plt.tight_layout()
plt.savefig('../../results/VAE/lrate/opt.png',dpi=150,format='png')
