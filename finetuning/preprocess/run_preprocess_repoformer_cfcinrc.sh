#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

SEQ_LENGTH=2048
MAX_CFC_LENGTH=512
LC_RC_RATIO=2.0

DATA_DIR="your_data_dir_here"
OUTPUT_DIR="${DATA_DIR}/processed/"

mkdir -p $OUTPUT_DIR

python preprocess_repoformer.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --cfc_in_rc \
    --seed 42 \
    --test_and_valid_combined_size 0.05 \
    --seq_length $SEQ_LENGTH \
    --max_cfc_length $MAX_CFC_LENGTH \
    --lc_rc_ratio 2.0 \
    --num_proc 40
