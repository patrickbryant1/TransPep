TRAIN_DATA=../data/train_set.fasta
VARIABLE_PARAMS=./param_combos.csv
OUTDIR=../results/

for PARAM_COMBO in {1..243}
do
  for TEST_PARTITION in {0..4}
    do
    ./transformer.py --train_data $TRAIN_DATA --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
    done
done
