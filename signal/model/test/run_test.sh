#!/usr/bin/env bash

CHECKPOINTDIR=../../checkpoint/
DATADIR=../../data/
BENCH_SET=../../data/benchmark_set.fasta
OUTDIR=../../results/

python3 nested_test.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --bench_set $BENCH_SET --outdir $OUTDIR
