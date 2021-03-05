#!/usr/bin/env bash


CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
DATADIR=../data/
OUTDIR=../results/

for TEST_PARTITION in {0..4}
  do
    JSONFILE=$CHECKPOINTDIR'TP'$TEST_PARTITION'/model.json'
    ./run_trained.py --json_file $JSONFILE --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
  done
