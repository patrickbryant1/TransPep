#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import glob
import pandas as pd
import numpy as np


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

            if get_seq == True:
                Seqs.appendd(line.rstrip())
                get_seq = False
                #Get annotation next
                get_annotation = True

            if get_annotation == True:


                get_annotation = False


    return data
