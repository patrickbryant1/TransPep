#!/usr/bin/env bash

VARIABLE_PARAMS=../param_combos.csv
CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/
DATADIR=../../data/
OUTDIR=../../results/


#0
TEST_PARTITION=0
PARAM_COMBO=46
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#1
TEST_PARTITION=1
PARAM_COMBO=47
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#2
TEST_PARTITION=2
PARAM_COMBO=48
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#3
TEST_PARTITION=3
PARAM_COMBO=49
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#4
TEST_PARTITION=4
PARAM_COMBO=50
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
