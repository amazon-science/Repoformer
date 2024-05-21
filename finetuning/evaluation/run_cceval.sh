#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export model_name='/home/diwu/code/Repoformer/paper/202309_repoformer/models/repoformer_1b_300k'
export exp=rcfcl_rg1  # baseline, rg1, oracle, lrcontext, rcfcl_rg1
export ranker=bm25

HOME_DIR=`realpath ../..`
data_root=`realpath ${HOME_DIR}/cceval/processed_data`

mkdir -p ${HOME_DIR}/results/cceval
output_root=${HOME_DIR}/results/cceval

# don't forget to activate proj_cc conda environment

# helpful command if we terminate jobs
# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 sudo kill -9
# ps -fA | grep python3 | awk '{ print $2 }' | xargs -n1 sudo kill -9

model_type=codelm
if [[ $exp == "rg1" || $exp == "oracle" || $exp == "repocoder" ]]; then
    model_type=codelm_cfc
elif [[ $exp == "lrcontext" ]]; then
    model_type=codelm_leftright_context
elif [[ $exp == "rcfcl_rg1" || $exp == "rcfcl_rg1_lrquery" || $exp == "rcfcl_repocoder" || $exp == "rcfcl_oracle" ]]; then
    model_type=codelm_right_cfc_left
elif [[ $exp == "cfcrl_rg1" || $exp == "cfcrl_repocoder" || $exp == "cfcrl_oracle" ]]; then
    model_type=codelm_cfc_right_left
fi

max_seq_length=2048

function run() {
    task=$1
    language=$2

    if [[ $exp == "baseline" || $exp == "lrcontext" ]]; then
        prompt_file="$data_root/${language}/${task}.jsonl"
        output_dir=$output_root/${language}/$exp/$task
    elif [[ $exp == "rcfcl_rg1" || $exp == "cfcrl_rg1" ]]; then
        prompt_file="$data_root/${language}/${task}_rg1_${ranker}.jsonl"
        output_dir=$output_root/${language}/$exp/$ranker/$task
    elif [[ $exp == "rcfcl_oracle" || $exp == "cfcrl_oracle" ]]; then
        prompt_file="$data_root/${language}/${task}_oracle_${ranker}.jsonl"
        output_dir=$output_root/${language}/$exp/$ranker/$task
    else
        prompt_file="$data_root/$setting/${language}/${task}_${exp}_${ranker}.jsonl"
        output_dir=$output_root/${language}/$setting/$exp/$ranker/$task
    fi

    # out_dirname=$(echo $model_name | tr '[:upper:]' '[:lower:]' | tr '/-' '_')
    out_dirname=repoformer_1b_threshold${selective_threshold}/
    output_dir=$output_dir/$out_dirname
    mkdir -p $output_dir

    gen_length=50
    if [[ $task == "function_completion" ]]; then
        gen_length=256
    fi

    if [[ $task == "chunk_completion" ]]; then
        task="api_completion"
    fi

    # nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 sudo kill -9
    # ps -fA | grep 'python3' | awk '{ print $2 }' | xargs -n1 sudo kill -9
    python ${HOME_DIR}/repo_eval/eval_vllm_repoformer.py \
        --task $task \
        --selective_retrieval \
	    --retrieval_threshold ${selective_threshold} \
        --compute_cceval_metric \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --cfc_seq_length 512 \
        --prompt_file $prompt_file \
        --gen_length $gen_length \
        --max_seq_length $max_seq_length \
        --output_dir $output_dir \
        --ts_lib ${HOME_DIR}/build/${language}-lang-parser.so \
        --language ${language} 2>&1 | tee $output_dir/log.txt
}

export selective_threshold=0.2
for lang in python java csharp typescript; do
    run line_completion $lang
done
