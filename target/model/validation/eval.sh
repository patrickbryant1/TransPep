#!/usr/bin/env bash

#Evaluate the loss
DATADIR=../../data/
RESULTSDIR=../../results/
VARIABLE_PARAMS=../param_combos.csv
#./eval_loss.py --resultsdir $RESULTSDIR --datadir $DATADIR --variable_params $VARIABLE_PARAMS


#Run the trained models
VARIABLE_PARAMS=../param_combos.csv
CHECKPOINTDIR=../../results/checkpoint/
DATADIR=../../data/
OUTDIR=../../results/


#0
TEST_PARTITION=0
PARAM_COMBO=691
#./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#1
TEST_PARTITION=1
PARAM_COMBO=837
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#2
TEST_PARTITION=2
PARAM_COMBO=638
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#3
TEST_PARTITION=3
PARAM_COMBO=564
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#4
TEST_PARTITION=4
PARAM_COMBO=845
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
