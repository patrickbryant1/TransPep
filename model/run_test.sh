#!/usr/bin/env bash

CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
DATADIR=../data/
OUTDIR=../results/

python3 test_model.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --outdir $OUTDIR
