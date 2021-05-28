
import pdb
import pandas as pd
import numpy as np
###########FUNTIONS###########
def parse_and_format(filename,data):
    '''Parse and format the data:
    '''

    AMINO_ACIDS = { 'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,
                'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,
                'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,
                'Y':18,'V':19,'X':20,'U':20, 'Z':20, 'B':20
              }


    IDs = []
    Seqs = []
    #Go through each line
    with open(filename) as file:
        for line in file:
            line = line.rstrip()
            if line[0]=='>': #Id
                IDs.append(line[1:])

            else: #Sequence
                current_seq = []
                for char in line:
                    current_seq.append(AMINO_ACIDS[char])
                #Pad or cut
                if len(current_seq)>200:
                    current_seq = current_seq[:200]
                else:
                    pad = np.zeros(200)
                    pad[:] = 20 #Unknown/unusual
                    pad[:len(current_seq)]=current_seq
                    current_seq = pad
                Seqs.append(current_seq)


    fasta_df = pd.DataFrame()
    fasta_df['ID'] = IDs
    fasta_df['Sequence'] = Seqs

    #Get the partitions and annotations from data
    ['x', 'y_cs', 'y_type', 'len_seq', 'org', 'fold', 'ids']
    #CS
    CS = data['y_cs']
    CS_num = np.zeros(len(CS)) #Number in sequence
    CS_pos = np.argwhere(CS==1) #Which seqs have CS and where
    CS_num[CS_pos[:,0]]=CS_pos[:,1] #Assign
    #Types
    Types = data['y_type']
    Orgs = data['org']
    Folds = data['fold']
    IDs = data['ids']
    #Annotation df
    annotation_df = pd.DataFrame()
    annotation_df['CS'] = CS_num
    annotation_df['Type'] = Types
    annotation_df['Org'] = Orgs
    annotation_df['Fold'] = Folds
    annotation_df['ID'] = IDs
    #Merge
    merged = pd.merge(annotation_df,fasta_df,on='ID')

    #Create annotations
    #5 classes of transit peptides
    #0=no targeting peptide, 1=sp: signal peptide, 2=mt:mitochondrial transit peptide,
    #3=ch:chloroplast transit peptide, 4=th:thylakoidal lumen composite transit peptide
    annotations = np.zeros((len(merged),200))
    for i in range(len(merged)):
        row = merged.loc[i]
        annotations[i,:int(row.CS)]=row.Type


    #Get sequences
    Seqs = []
    [Seqs.append(np.array(seq)) for seq in merged.Sequence]

    return merged, np.array(Seqs), annotations
