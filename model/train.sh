TRAIN_DATA=../data/train_set.fasta
VARIABLE_PARAMS=./variable_params.csv
PARAM_COMBO=1
OUTDIR=../results/

for PARTITION in {0..4}
do
./transformer.py --train_data $TRAIN_DATA --partition $PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
done
