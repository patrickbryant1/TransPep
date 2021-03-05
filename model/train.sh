TRAIN_DATA=../data/train_set.fasta
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
SAVE_MODEL=1
CHECKPOINT=1
NUM_EPOCHS=50
OUTDIR=../results/

#0
TEST_PARTITION=0
PARAM_COMBO=208
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#1
TEST_PARTITION=1
PARAM_COMBO=208
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#2
TEST_PARTITION=2
PARAM_COMBO=118
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#3
TEST_PARTITION=3
PARAM_COMBO=110
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#3
TEST_PARTITION=4
PARAM_COMBO=119
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR
