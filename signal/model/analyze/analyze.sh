#!/usr/bin/env bash

CHECKPOINTDIR=../../checkpoint/
DATADIR=../../data/
BENCH_SET=../../data/benchmark_set.fasta
OUTDIR=/home/patrick/results/protein_translation/attentions/

TP=0
for VP in 1 2 3 4
do
  mkdir $OUTDIR'/TP'$TP'/VP'$VP
  python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP &> $OUTDIR'/TP'$TP'/VP'$VP'/activations.txt' 
done
