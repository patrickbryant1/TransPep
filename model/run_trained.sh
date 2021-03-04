#!/usr/bin/env bash

JSONFILE='/home/patrick/results/protein_translation/model_checkpoints/model.json'
WEIGHTS=/home/patrick/results/protein_translation/model_checkpoints/weights-03-.hdf5
DATADIR=../data/
OUTDIR=../results/

for TEST_PARTITION in {0..4}
  do
    ./run_trained.py --json_file $JSONFILE --weights $WEIGHTS --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
  done
