#!/usr/bin/env bash

#Evaluate the loss
DATADIR=../../data/
RESULTSDIR=../../results/focus/
VARIABLE_PARAMS=../param_combos.csv
./eval_loss.py --resultsdir $RESULTSDIR --datadir $DATADIR --variable_params $VARIABLE_PARAMS

#Evaluate the validation
RESULTSDIR=../../results/focus/
./eval_valid.py --resultsdir $RESULTSDIR
