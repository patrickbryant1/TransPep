
JSONFILE=
WEIGHTS=/home/patrick/results/protein_translation/model_checkpoints/weights-01-.hdf5
DATADIR=../data/

for TEST_PARTITION in {0..4}
  do
  python3 ./run_trained.sh --json_file $JSONFILE --weights $WEIGHTS --datadir $DATADIR --test_partition $TEST_PARTITION --variable_params $VARIABLE_PARAMS --param_combo $PARAM_COMBO --checkpointdir $CHECKPOINTDIR --outdir $OUTDIR
  done
