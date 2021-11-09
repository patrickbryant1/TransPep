


import argparse
import sys
import os
import numpy as np
import pandas as pd
import pdb
#Arguments for argparse module:
parser = argparse.ArgumentParser(description = '''Read a csv and write the table into LateX format.''')
parser.add_argument('--csv', nargs=1, type= str, default=sys.stdin, help = 'Path to table in csv.')


######################MAIN######################
args = parser.parse_args()
csv = pd.read_csv(args.csv[0])

cols = csv.columns

#Print LateX
print('\\begin{tabular}{'+'l'*len(cols)+'}')
print('\\toprule')
print('&'.join(cols)+'\\'+'\\')
print('\\midrule')
for i in range(len(csv)):
    row = csv.loc[i]
    vals = [*row.values]
    str_row = []
    for j in range(len(vals)):
        try:
            str_row.append(str(np.round(float(vals[j]),3)))
        except:
            str_row.append(vals[j])

    print('&'.join(str_row)+'\\'+'\\')
print('\\hline')
print('\\end{tabular}')

'''
\begin{tabular}{llllll}
\toprule
 Type&Archaea&Eukaryotes&Gram-negative bacteria&Gram-positive bacteria\\
\midrule
Sec/SPI&60 (50)&2614 (210)&509 (90)&189 (25)\\
Sec/SPII&28 (19)&N.A.&1063 (442)&449 (201)\\
Tat/SPI&27 (22)&N.A.&334 (98)&95 (74)\\
Globular&78 (63)&13612 (6929)&202 (103)&140 (64)\\
Membrane&44 (28)&1044 (318)&220 (50)6&50 (25)\\
Total&237 (182)&17270 (7457)&2328 (783)&923 (389)\\

 \hline
\end{tabular}
'''
