#!/usr/bin/env bash

VARIABLE_PARAMS=../param_combos.csv
CHECKPOINTDIR=/home/patrick/results/protein_translation/model_checkpoints/focus/
DATADIR=../../data/
OUTDIR=../../results/

#Focus
#0
TEST_PARTITION=0
PARAM_COMBO=1021
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#1
TEST_PARTITION=1
PARAM_COMBO=992
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#2
TEST_PARTITION=2
PARAM_COMBO=723
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#3
TEST_PARTITION=3
PARAM_COMBO=274
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR

#4
TEST_PARTITION=4
PARAM_COMBO=815
./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR


# #0
# TEST_PARTITION=0
# PARAM_COMBO=886
# ./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
#
# #1
# TEST_PARTITION=1
# PARAM_COMBO=737
# ./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
#
# #2
# TEST_PARTITION=2
# PARAM_COMBO=273
# ./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
#
# #3
# TEST_PARTITION=3
# PARAM_COMBO=274
# ./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
#
# #4
# TEST_PARTITION=4
# PARAM_COMBO=275
# ./run_trained.py --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR'TP'$TEST_PARTITION'/' --datadir $DATADIR --test_partition $TEST_PARTITION --outdir $OUTDIR
