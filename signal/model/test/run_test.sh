#!/usr/bin/env bash

VARIABLE_PARAMS=../param_combos.csv
PARAM_COMBOS=886,737,273,274,275
CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
DATADIR=../../data/
BENCH_SET=../../data/benchmark_set.fasta
OUTDIR=../../results/

python3 nested_test.py --variable_params $VARIABLE_PARAMS --param_combos $PARAM_COMBOS --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --bench_set $BENCH_SET --outdir $OUTDIR
