
import pdb
import pandas as pd
import numpy as np
###########FUNTIONS###########
def parse_and_format(filename,data):
    '''Parse and format the data:
    '''

    AMINO_ACIDS = { 'A':1,'R':2,'N':3,'D':4,'C':5,'E':6,
                'Q':7,'G':8,'H':9,'I':10,'L':11,'K':12,
                'M':13,'F':14,'P':15,'S':16,'T':17,'W':18,
                'Y':19,'V':20,'X':21,'U':21, 'Z':21, 'B':21
              }


    IDs = []
    Seqs = []
    Lens = []
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
                #Seqlen
                Lens.append(len(current_seq))
                #Pad or cut
                if len(current_seq)>200:
                    current_seq = current_seq[:200]
                else:
                    pad = np.zeros(200)
                    pad[:len(current_seq)]=current_seq
                    current_seq = pad
                #Seq
                Seqs.append(current_seq)



    fasta_df = pd.DataFrame()
    fasta_df['ID'] = IDs
    fasta_df['Sequence'] = Seqs
    fasta_df['Seqlen'] = Lens

    #Get the partitions and annotations from data
    ['x', 'y_cs', 'y_type', 'len_seq', 'org', 'fold', 'ids']
    #CS
    CS = data['y_cs']
    CS_num = np.zeros(len(CS)) #Number in sequence
    CS_pos = np.argwhere(CS==1) #Which seqs have CS and where
    CS_num[CS_pos[:,0]]=CS_pos[:,1] #Assign
    #Types
    Types = data['y_type']+1
    Orgs = data['org']
    Folds = data['fold']
    IDs = data['ids']
    #Annotation df
    annotation_df = pd.DataFrame()
    annotation_df['CS'] = CS_num #These are zero indexed, meaning that e.g. pos 45 is actually pos 46  - which will be the last position with a TP
    annotation_df['Type'] = Types
    annotation_df['Org'] = Orgs
    annotation_df['Fold'] = Folds
    annotation_df['ID'] = IDs
    #Merge
    merged = pd.merge(annotation_df,fasta_df,on='ID')

    #Create annotations
    #5 classes of transit peptides
    #0 = pad
    #1=no targeting peptide/Inside cell, 2=sp: signal peptide, 3=mt:mitochondrial transit peptide,
    #4=ch:chloroplast transit peptide, 5=th:thylakoidal lumen composite transit peptide
    #6=Outside of cell - only valid for SPs - not for the peptides going into mt or ch/th
    annotations = np.zeros((len(merged),200))
    for i in range(len(merged)):
        row = merged.loc[i]
        annotations[i,:int(row.CS)+1]=row.Type

        #Inside of cell/organoid - non secreted
        if row.Type != 2:
            annotations[i,int(row.CS)+1:row.Seqlen]=1
        #SP
        else:
            annotations[i,int(row.CS)+1:row.Seqlen]=6
        pdb.set_trace()


    #Get sequences
    Seqs = []
    [Seqs.append(np.array(seq)) for seq in merged.Sequence]
    #Drop sequences from merged
    merged = merged.drop(columns=['Sequence'])

    return merged, np.array(Seqs), annotations
