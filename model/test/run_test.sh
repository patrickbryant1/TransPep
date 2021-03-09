#!/usr/bin/env bash

CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
DATADIR=../../data/
BENCH_SET=../../data/benchmark_set.fasta
OUTDIR=../../results/

python3 nested_test.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --bench_set $BENCH_SET --outdir $OUTDIR
