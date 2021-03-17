#!/usr/bin/env bash

CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
DATADIR=../../data/
BENCH_SET=../../data/benchmark_set.fasta
VARIABLE_PARAMS=../param_combos.csv
OUTDIR=../../results/attention/

# TP=0
# PARAM_COMBO=886
# for VP in 1 2 3 4
# do
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
# done
#
# TP=1
# mkdir $OUTDIR'/TP'$TP
# for VP in 0 2 3 4
# do
#   mkdir $OUTDIR'/TP'$TP'/VP'$VP
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP &> $OUTDIR'/TP'$TP'/VP'$VP'/activations.txt'
# done
#
# TP=2
# mkdir $OUTDIR'/TP'$TP
# for VP in 0 1 3 4
# do
#   mkdir $OUTDIR'/TP'$TP'/VP'$VP
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP &> $OUTDIR'/TP'$TP'/VP'$VP'/activations.txt'
# done
#
# TP=3
# mkdir $OUTDIR'/TP'$TP
# for VP in 0 1 2 4
# do
#   mkdir $OUTDIR'/TP'$TP'/VP'$VP
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP &> $OUTDIR'/TP'$TP'/VP'$VP'/activations.txt'
# done
#
# TP=4
# mkdir $OUTDIR'/TP'$TP
# for VP in 0 1 2 3
# do
#   mkdir $OUTDIR'/TP'$TP'/VP'$VP
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP &> $OUTDIR'/TP'$TP'/VP'$VP'/activations.txt'
# done


#Analyze attention
TP=0
ATTENTIONDIR=$OUTDIR
python3 analyze_attention.py --attention_dir $ATTENTIONDIR --test_partition $TP
