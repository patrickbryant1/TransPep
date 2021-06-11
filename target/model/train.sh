
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
CHECKPOINTDIR=../results/checkpoint/
CHECKPOINT=1
NUM_EPOCHS=20
FINDLR=0
OUTDIR=../results/

PARAM_COMBO=186
#TEST_PARTITION=0
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --find_lr $FINDLR --outdir $OUTDIR

PARAM_COMBO=302
#TEST_PARTITION=1
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --find_lr $FINDLR --outdir $OUTDIR

PARAM_COMBO=183
#TEST_PARTITION=2
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --find_lr $FINDLR --outdir $OUTDIR

PARAM_COMBO=189
#TEST_PARTITION=3
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --find_lr $FINDLR --outdir $OUTDIR

PARAM_COMBO=245
#TEST_PARTITION=4
python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --find_lr $FINDLR --outdir $OUTDIR
