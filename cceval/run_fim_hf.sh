#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export model=${1:-"starcoderbase-1b"}
export exp=${2:-"rcfcl_rg1"}   # baseline, rg1, oracle
export ranker=${3:-"bm25"}     # bm25 openai_cosine_sim unixcoder_cosine_sim

HOME_DIR=`realpath ..`
data_root=`realpath ./processed_data`
mkdir -p ${HOME_DIR}/results/cceval
output_root=${HOME_DIR}/results/cceval


# You may use other non-fim models. To do so, simply provide the model name and remove the "--use_fim_prompt" flag.
# Also, specify the batch size and dtype.
declare -A py_model_zoo
py_model_zoo["starcoder"]="bigcode/starcoder"
py_model_zoo["starcoderbase"]="bigcode/starcoderbase"
py_model_zoo["starcoderbase-7b"]="bigcode/starcoderbase-7b"
py_model_zoo["starcoderbase-3b"]="bigcode/starcoderbase-3b"
py_model_zoo["starcoderbase-1b"]="bigcode/starcoderbase-1b"

declare -A batch_size
batch_size["starcoder"]=1
batch_size["starcoderbase"]=1
batch_size["starcoderbase-7b"]=4 
batch_size["starcoderbase-3b"]=8 
batch_size["starcoderbase-1b"]=8 

# helpful command if we terminate jobs
# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 sudo kill -9
# ps -fA | grep python3 | awk '{ print $2 }' | xargs -n1 sudo kill -9

model_name=${py_model_zoo["$model"]}
model_type=codelm
if [[ $exp == "rg1" || $exp == "oracle" ]]; then
    model_type=codelm_cfc
elif [[ $exp == "lrcontext" ]]; then
    model_type=codelm_leftright_context
elif [[ $exp == "rcfcl_rg1" || $exp == "rcfcl_oracle" ]]; then
    model_type=codelm_right_cfc_left
fi

max_seq_length=2048
dtype=bf16

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

    out_dirname=$(echo $model_name | tr '[:upper:]' '[:lower:]' | tr '/-' '_')
    output_dir=$output_dir/$out_dirname
    mkdir -p $output_dir

    gen_length=50
    if [[ $task == "function_completion" ]]; then
        gen_length=256
    fi

    if [[ $task == "chunk_completion" ]]; then
        task="api_completion"
    fi

    accelerate launch --main_process_port 29570 ${HOME_DIR}/repo_eval/eval_hf.py \
        --task $task \
        --compute_cceval_metric \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --use_fim_prompt \
        --preprocessing_num_workers 1 \
        --cfc_seq_length 512 \
        --min_cfc_score 0.0 \
        --prompt_file $prompt_file \
        --gen_length $gen_length \
        --max_seq_length $max_seq_length \
        --preprocessing_num_workers 1 \
        --batch_size ${batch_size["$model"]} \
        --output_dir $output_dir \
        --dtype $dtype \
        --ts_lib ${HOME_DIR}/build/${language}-lang-parser.so \
        --language ${language} 2>&1 | tee $output_dir/log.txt
}

for lang in python java csharp typescript; do
    run line_completion $lang
done
