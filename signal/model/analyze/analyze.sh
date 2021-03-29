#!/usr/bin/env bash

CHECKPOINTDIR=../../checkpoint/
DATADIR=../../data/
BENCH_SET=../../data/benchmark_set.fasta
VARIABLE_PARAMS=../param_combos.csv
OUTDIR=/home/patrick/results/protein_translation/attention/

# TP=0
# PARAM_COMBO=886
# for VP in 1 2 3 4
# do
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
# done
#
# TP=1
# PARAM_COMBO=737
# for VP in 0 2 3 4
# do
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
# done
#
# TP=2
# PARAM_COMBO=273
# for VP in 0 1 3 4
# do
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
# done
#
# TP=3
# PARAM_COMBO=274
# for VP in 0 1 2 4
# do
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
# done
#
# TP=4
# PARAM_COMBO=275
# for VP in 0 1 2 3
# do
#   python3 get_attention.py --checkpointdir $CHECKPOINTDIR --datadir $DATADIR --test_partition $TP --valid_partition $VP --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --outdir $OUTDIR
# done


#Analyze attention
ATTENTIONDIR=$OUTDIR
#python3 analyze_attention.py --attention_dir $ATTENTIONDIR

#Search motifs
#kingdom_conversion = {'ARCHAEA':0,'EUKARYA':1,'NEGATIVE':2,'POSITIVE':3}
NMOTIF='(R|K)(D|T|M|R)(K|S|M|Q)'
CMOTIF='(I|F|V|T|S|A).(S|A)'
python3 search_motifs.py --datadir $DATADIR --kingdom 0 --type 'SP' --nmotif $NMOTIF --cmotif $CMOTIF

#Analyze the benchmark results
BENCHCSV=../../results/no_evolution/benchmark.csv
OUTDIR=../../results/figures/
#python3 plot_bench.py --benchcsv $BENCHCSV --outdir $OUTDIR
