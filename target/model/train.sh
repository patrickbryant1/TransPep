
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
SAVE_MODEL=1
CHECKPOINT=1
NUM_EPOCHS=50
OUTDIR=../results/

TEST_PARTITION=0
python3 ./transformer.py --datadir $DATADIR --test_partition $TEST_PARTITION --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR
