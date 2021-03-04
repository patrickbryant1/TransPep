#!/usr/bin/env bash

DATADIR=../data/
RESULTSDIR=../results/
VARIABLE_PARAMS=./param_combos.csv
./eval.py --resultsdir $RESULTSDIR --datadir $DATADIR --variable_params $VARIABLE_PARAMS
