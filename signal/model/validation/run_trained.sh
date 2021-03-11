#!/usr/bin/env bash

VARIABLE_PARAMS=../param_combos.csv
PARAM_COMBO=208
CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
DATADIR=../../data/
OUTDIR=../../results/

for TEST_PARTITION in {0..4}
  do
    JSONFILE=$CHECKPOINTDIR'TP'$TEST_PARTITION'/model.json'
    ./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
  done
