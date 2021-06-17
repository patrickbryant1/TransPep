
DATADIR=../data/
VARIABLE_PARAMS=./param_combos.csv
CHECKPOINTDIR=../results/checkpoint/
CHECKPOINT=0
NUM_EPOCHS=1
FINDLR=1
OUTDIR=../results/lrate/

for PARAM_COMBO in {1..360}
do
  echo $PARAM_COMBO
  python3 ./transformer.py --datadir $DATADIR --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'/' --checkpoint $CHECKPOINT --num_epochs $NUM_EPOCHS --find_lr $FINDLR --outdir $OUTDIR
done
