#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8
data_dir=`realpath ../processed_data`

function prepare_sparse_retrieved_data() {
    for task in line api "function"; do

        if [ ${task} == "line" ]; then
            CHUNK_SIZE=10
            SLIDING_WINDOW_SIZE=5
            QUERY_LENGTH=10
            TOPK=10
            echo "${task} sparse retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        elif [ ${task} == "api" ]; then
            CHUNK_SIZE=10
            SLIDING_WINDOW_SIZE=5
            QUERY_LENGTH=10
            TOPK=10
            echo "${task} sparse retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        else
            CHUNK_SIZE=20
            SLIDING_WINDOW_SIZE=10
            QUERY_LENGTH=20
            TOPK=5
            echo "${task} sparse retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        fi
	
        python attach_cfc.py \
            --input_file $data_dir/python_${task}_completion.jsonl \
            --repository_file $data_dir/repocoder_packages.jsonl \
	        --chunk_size ${CHUNK_SIZE} --sliding_window_size ${SLIDING_WINDOW_SIZE} --query_length ${QUERY_LENGTH} --use_topk_chunks ${TOPK} \
	        --ranking_fn jaccard_similarity \
            --query_type last_n_lines \
            --crossfile_distance 100 \
            --maximum_chunk_to_rerank 1000 \
            --maximum_cross_files 1000 \
            --output_file_suffix "sparse_rg1"

        python attach_cfc.py \
            --input_file $data_dir/python_${task}_completion.jsonl \
            --repository_file $data_dir/repocoder_packages.jsonl \
            --chunk_size ${CHUNK_SIZE} --sliding_window_size ${SLIDING_WINDOW_SIZE} --query_length ${QUERY_LENGTH} --use_topk_chunks ${TOPK} \
            --ranking_fn jaccard_similarity \
            --query_type groundtruth \
            --crossfile_distance 100 \
            --maximum_chunk_to_rerank 1000 \
            --maximum_cross_files 1000 \
            --output_file_suffix "sparse_oracle"
    done
}

function prepare_sparse_codebleu_retrieved_data() {
    for task in line api "function"; do

        if [ ${task} == "line" ]; then
            CHUNK_SIZE=10
            SLIDING_WINDOW_SIZE=5
            QUERY_LENGTH=10
            TOPK=10
            echo "${task} codebleu retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        elif [ ${task} == "api" ]; then
            CHUNK_SIZE=10
            SLIDING_WINDOW_SIZE=5
            QUERY_LENGTH=10
            TOPK=10
            echo "${task} codebleu retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        else
            CHUNK_SIZE=20
            SLIDING_WINDOW_SIZE=10
            QUERY_LENGTH=20
            TOPK=5
            echo "${task} codebleu retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        fi

        python attach_cfc.py \
            --input_file $data_dir/python_${task}_completion.jsonl \
            --repository_file $data_dir/repocoder_packages.jsonl \
            --chunk_size ${CHUNK_SIZE} --sliding_window_size ${SLIDING_WINDOW_SIZE} --query_length ${QUERY_LENGTH} --use_topk_chunks ${TOPK} \
            --ranking_fn weighted_ngram_match_score \
            --query_type last_n_lines \
            --crossfile_distance 100 \
            --maximum_chunk_to_rerank 1000 \
            --maximum_cross_files 1000 \
            --output_file_suffix "sparse_wngrammatch_rg1"

        python attach_cfc.py \
            --input_file $data_dir/python_${task}_completion.jsonl \
            --repository_file $data_dir/repocoder_packages.jsonl \
            --chunk_size ${CHUNK_SIZE} --sliding_window_size ${SLIDING_WINDOW_SIZE} --query_length ${QUERY_LENGTH} --use_topk_chunks ${TOPK} \
            --ranking_fn weighted_ngram_match_score \
            --query_type groundtruth \
            --crossfile_distance 100 \
            --maximum_chunk_to_rerank 1000 \
            --maximum_cross_files 1000 \
            --output_file_suffix "sparse_wngrammatch_oracle"
    done
}

function prepare_unixcoder_retrieved_data() {
    for task in line api "function"; do

        if [ ${task} == "line" ]; then
            CHUNK_SIZE=10
            SLIDING_WINDOW_SIZE=5
            QUERY_LENGTH=10
            TOPK=10
            echo "${task} unixcoder retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        elif [ ${task} == "api" ]; then
            CHUNK_SIZE=10
            SLIDING_WINDOW_SIZE=5
            QUERY_LENGTH=10
            TOPK=10
            echo "${task} unixcoder retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        else
            CHUNK_SIZE=20
            SLIDING_WINDOW_SIZE=10
            QUERY_LENGTH=20
            TOPK=5
            echo "${task} unixcoder retrieval CHUNK SIZE=${CHUNK_SIZE} SLIDING WINDOW SIZE=${SLIDING_WINDOW_SIZE} QUERY LENGTH=${QUERY_LENGTH} TOPK=${TOPK}"
        fi

        python attach_cfc.py \
            --input_file $data_dir/python_${task}_completion.jsonl \
            --repository_file $data_dir/repocoder_packages.jsonl \
            --chunk_size ${CHUNK_SIZE} --sliding_window_size ${SLIDING_WINDOW_SIZE} --query_length ${QUERY_LENGTH} --use_topk_chunks ${TOPK} \
            --ranking_fn cosine_similarity \
            --dense_retriever_type unixcoder \
            --query_type last_n_lines \
            --crossfile_distance 100 \
            --maximum_chunk_to_rerank 1000 \
            --maximum_cross_files 1000 \
            --output_file_suffix "unixcoder_rg1"

        python attach_cfc.py \
            --input_file $data_dir/python_${task}_completion.jsonl \
            --repository_file $data_dir/repocoder_packages.jsonl \
            --chunk_size ${CHUNK_SIZE} --sliding_window_size ${SLIDING_WINDOW_SIZE} --query_length ${QUERY_LENGTH} --use_topk_chunks ${TOPK} \
            --ranking_fn cosine_similarity \
            --dense_retriever_type unixcoder \
            --query_type groundtruth \
            --crossfile_distance 100 \
            --maximum_chunk_to_rerank 1000 \
            --maximum_cross_files 1000 \
            --output_file_suffix "unixcoder_oracle"
    done
}


prepare_sparse_retrieved_data
# prepare_unixcoder_retrieved_data
# prepare_sparse_codebleu_retrieved_data
