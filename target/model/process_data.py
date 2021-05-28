
import pdb
import pandas as pd
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
                Seqs.append(current_seq)


    fasta_df = pd.DataFrame()
    fasta_df['ID'] = IDs
    fasta_df['Sequence'] = Seqs

    #Get the partitions and annotations from data
    ['x', 'y_cs', 'y_type', 'len_seq', 'org', 'fold', 'ids']
    CS = data['y_cs']
    Types = data['y_type']
    Orgs = data['org']
    Folds = data['fold']
    IDs = data['ids']
    annotation_df = pd.DataFrame()
    annotation_df['CS'] = CS
    pdb.set_trace()
