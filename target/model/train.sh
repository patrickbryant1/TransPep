
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
CHECKPOINTDIR=../results/checkpoint/
SAVE_MODEL=1
CHECKPOINT=1
NUM_EPOCHS=30
OUTDIR=../results/

PARAM_COMBO=691
TEST_PARTITION=0
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

PARAM_COMBO=837
TEST_PARTITION=1
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

PARAM_COMBO=638
TEST_PARTITION=2
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

PARAM_COMBO=564
TEST_PARTITION=3
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR

PARAM_COMBO=845
TEST_PARTITION=4
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --outdir $OUTDIR
