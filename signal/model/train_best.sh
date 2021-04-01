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
PARAM_COMBO=886
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#1
TEST_PARTITION=1
PARAM_COMBO=737
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#2
TEST_PARTITION=2
PARAM_COMBO=273
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR  --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#3
TEST_PARTITION=3
PARAM_COMBO=274
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

#4
TEST_PARTITION=4
PARAM_COMBO=275
python3 ./transformer.py --train_data $TRAIN_DATA --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --save_model $SAVE_MODEL --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR
