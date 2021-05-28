
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
CHECKPOINTDIR=../results/checkpoint/
SAVE_MODEL=1
CHECKPOINT=1
NUM_EPOCHS=5
OUTDIR=../results/

PARAM_COMBO=1
TEST_PARTITION=0
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR
