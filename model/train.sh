TRAIN_DATA=../data/train_set.fasta
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
SAVE_MODEL=1
OUTDIR=../results/

for PARAM_COMBO in {1..243}
do
  for TEST_PARTITION in {0..4}
    do
    python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --outdir $OUTDIR
    done
done
