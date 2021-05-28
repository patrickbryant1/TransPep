

###########FUNTIONS###########
def parse_and_format(filename):
    '''Parse and format the data:
    '''

    AMINO_ACIDS = { 'A':0,'R':1,'N':2,'D':3,'C':4,'E':5,
                'Q':6,'G':7,'H':8,'I':9,'L':10,'K':11,
                'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,
                'Y':18,'V':19,'X':20
              }


    IDs = []
    Seqs = []
    #Go through each line
    with open(filename) as file:
        for line in file:
            if line[0]=='>': #Id
                IDs.append(line[1:].rstrip())

            else: #Sequence
                current_seq = []
                for char in line:
                    current_seq.append(AMINO_ACIDS[char])
                Seqs.append(current_seq)

    
    pdb.set_trace()
