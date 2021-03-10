#!/usr/bin/env bash

CHECKPOINTDIR=../../checkpoint/
DATADIR=../../data/
BENCH_SET=../../data/benchmark_set.fasta
OUTDIR=/home/patrick/results/protein_translation/attention/

# TP=0
# mkdir $OUTDIR'/TP'$TP
# for VP in 1 2 3 4
# do
#   mkdir $OUTDIR'/TP'$TP'/VP'$VP
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP &> $OUTDIR'/TP'$TP'/VP'$VP'/activations.txt'
# done

#Parse attention
TP=0
ATTENTIONDIR=$OUTDIR
#python3 parse_attention.py --attention_dir $ATTENTIONDIR --test_partition $TP

#Analyze attention
TP=0
ATTENTIONDIR=$OUTDIR
python3 analyze_attention.py --attention_dir $ATTENTIONDIR --checkpointdir $CHECKPOINTDIR'TP'$TP'/' --test_partition $TP
