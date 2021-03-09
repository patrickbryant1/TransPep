#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import glob
import pandas as pd
import numpy as np
from collections import Counter

#import matplotlib.pyplot as plt
#import seaborn as sns

import pdb



###########FUNTIONS###########
def parse_and_format(filename):
    '''Parse and format the data:
    >Uniprot_AC|Kingdom|Type|Partition No
    amino-acid sequence
    annotation [S: Sec/SPI signal peptide | T: Tat/SPI signal peptide | L: Sec/SPII signal peptide | I: cytoplasm | M: transmembrane | O: extracellular]

    >P36001|EUKARYA|NO_SP|1
    MDDISGRQTLPRINRLLEHVGNPQDSLSILHIAGTNGKETVSKFLTSILQHPGQQRQRVLIGRYTTSSLL
    IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII
    '''

    IDs = []
    Kingdoms = []
    Types = []
    Partitions = []
    Seqs = []
    Annotations = []
    #Conversions
    kingdom_conversion = {'ARCHAEA':0,'EUKARYA':1,'NEGATIVE':2,'POSITIVE':3}
    annotation_conversion = {'S':0,'T':1,'L':2,'I':3,'M':4,'O':5}
    AMINO_ACIDS = { 'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,
                    'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,
                    'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,
                    'Y':18,'V':19,'X':20
                  }
    #Keep track of what to get
    get_seq = False
    get_annotation = False
    #Go through each line
    with open(filename) as file:
        for line in file:
            if line[0]=='>':
                line = line[1:].rstrip()
                line = line.split('|')
                IDs.append(line[0])
                Kingdoms.append(kingdom_conversion[line[1]])
                Types.append(line[2])
                Partitions.append(line[3])
                #Get sequence next
                get_seq = True
                continue

            if get_seq == True:
                #Get sequence
                line = line.rstrip()
                if len(line)<70: #Add X if shorter than 70
                    line = line+'X'*(70-len(line))
                current_seq = []
                for char in line:
                    current_seq.append(AMINO_ACIDS[char])
                Seqs.append(current_seq)
                get_seq = False
                #Get annotation next
                get_annotation = True
                continue

            if get_annotation == True:
                #Get annotation
                line = line.rstrip()
                current_annotation = []
                for char in line:
                    current_annotation.append(annotation_conversion[char])

                #Check that the length is at least 70
                if len(current_annotation)<70:
                    new_annotation = np.zeros(70,dtype='int')
                    new_annotation[:len(current_annotation)]=current_annotation
                    new_annotation[len(current_annotation):]=current_annotation[-1]
                    current_annotation = new_annotation
                #Save
                Annotations.append(current_annotation)
                get_annotation = False


    data = pd.DataFrame()
    data['ID']=IDs
    data['Kingdom']=Kingdoms
    data['Type']=Types
    data['Partition']=Partitions
    Seqs = np.array(Seqs)
    Annotations = np.array(Annotations)
    #Get clevage sites
    CSs = []
    for i in range(len(Annotations)):

        if min(Annotations[i])<3:
            CSs.append(np.where(Annotations[i]<3)[0][-1])
        else:
            CSs.append(0)

    data['CS']=CSs

    return data, Seqs, Annotations