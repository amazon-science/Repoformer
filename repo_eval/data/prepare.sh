#!/usr/bin/env bash

output_dir=`realpath ../processed_data`
mkdir -p $output_dir

if [[ ! -f $output_dir/repocoder_packages.jsonl ]]; then
    python jsonify.py \
        --repo_dir repositories \
        --output_dir $output_dir
fi

python convert.py \
    --repo_dir repositories \
    --prompt_file datasets/line_level_completion_2k_context_codegen.test.jsonl \
    --output_file $output_dir/python_line_completion.jsonl \
    --repo_type line

python convert.py \
    --repo_dir repositories \
    --prompt_file datasets/api_level_completion_2k_context_codegen.test.jsonl \
    --output_file $output_dir/python_api_completion.jsonl \
    --repo_type line

python convert.py \
    --repo_dir repositories \
    --prompt_file datasets/function_level_completion_2k_context_codex.test.jsonl \
    --output_file $output_dir/python_function_completion.jsonl \
    --repo_type function
