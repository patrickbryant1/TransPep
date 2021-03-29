DATADIR=../../data/
#Search motifs
#kingdom_conversion = {'ARCHAEA':0,'EUKARYA':1,'NEGATIVE':2,'POSITIVE':3}
#######SEC/SPI#########
#ARCHAEA
#Attention motif
NMOTIF='(R|K),(K|R)'
CMOTIF='(I|F|V|T|S|A).(S|A)'
python3 search_motifs.py --datadir $DATADIR --kingdom 0 --type 'SP' --nmotif $NMOTIF --cmotif $CMOTIF
#Seq motif
NMOTIF='(A|N),(M|K)'
CMOTIF='(A|V|S).A'
python3 search_motifs.py --datadir $DATADIR --kingdom 0 --type 'SP' --nmotif $NMOTIF --cmotif $CMOTIF

#EUKARYA
#Attention motif
# NMOTIF='(R|M|L|K|F),(Y|K|P|M|F|S),(R|K|G|M|S)'
# CMOTIF='(L|T|N|V|D|W).(K|E|Q|L|C|Y|W)'
# python3 search_motifs.py --datadir $DATADIR --kingdom 0 --type 'SP' --nmotif $NMOTIF --cmotif $CMOTIF
# #Seq motif
# NMOTIF='A,A,A'
# CMOTIF='(A|V|S|T|C).(A|G|S)'
# python3 search_motifs.py --datadir $DATADIR --kingdom 0 --type 'SP' --nmotif $NMOTIF --cmotif $CMOTIF
