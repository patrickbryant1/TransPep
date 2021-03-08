#!/usr/bin/env bash

#Evaluate the loss
DATADIR=../data/
RESULTSDIR=../results/
VARIABLE_PARAMS=./param_combos.csv
#./eval_loss.py --resultsdir $RESULTSDIR --datadir $DATADIR --variable_params $VARIABLE_PARAMS

#Evaluate the validation
RESULTSDIR=../results/
./eval_valid.py --resultsdir $RESULTSDIR
