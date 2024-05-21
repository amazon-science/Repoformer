#!/usr/bin/env bash

export PYTHONIOENCODING=utf-8

export HOME_DIR=`realpath ../../../`

export model=$1
export outdir=$2
export language=$3
export port=${4:-29501}
export exp=lrcontext
export ranker=sparse

declare -A py_model_zoo
py_model_zoo["starcoder"]="bigcode/starcoder"
py_model_zoo["starcoderbase-7b"]="bigcode/starcoderbase-7b"
py_model_zoo["starcoderbase-3b"]="bigcode/starcoderbase-3b"
py_model_zoo["starcoderbase-1b"]="bigcode/starcoderbase-1b"
py_model_zoo["santacoder"]="bigcode/santacoder"

declare -A batch_size
batch_size["starcoder"]=2
batch_size["starcoderbase-7b"]=4
batch_size["starcoderbase-3b"]=8
batch_size["starcoderbase-1b"]=32
batch_size["santacoder"]=8

# don't forget to activate proj_cc conda environment

# helpful command if we terminate jobs
# nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 sudo kill -9
# ps -fA | grep python3 | awk '{ print $2 }' | xargs -n1 sudo kill -9

model_name=${py_model_zoo["$model"]}
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
dtype=fp16
if [[ $model == "starcoder" || $model == "starcoderbase-1b" || $model == "starcoderbase-3b" || $model == "starcoderbase-7b" ]]; then
    # max_seq_length=8192
    dtype=bf16
elif [[ $model == "santacoder" ]]; then
    dtype=fp32
elif [[ $model == "santacoder_no_fim" ]]; then
    dtype=fp32
fi

function run() {
    task=$1

    prompt_file="${outdir}/sample_for_completion.jsonl"
    output_dir="${outdir}/logs/lrcontext/"

    out_dirname=$model #$(echo $model_name | tr '[:upper:]' '[:lower:]' | tr '/-' '_')
    output_dir=$output_dir/$out_dirname
    mkdir -p $output_dir

    gen_length=50
    if [[ $task == "function_completion" ]]; then
        gen_length=256
    fi

    # nvidia-smi | grep 'python' | awk '{ print $5 }' | xargs -n1 sudo kill -9
    # ps -fA | grep 'python3' | awk '{ print $2 }' | xargs -n1 sudo kill -9
    accelerate launch --main_process_port ${port} ${HOME_DIR}/repo_eval/eval_hf.py \
        --task $task \
        --model_type $model_type \
        --model_name_or_path $model_name \
        --use_fim_prompt \
        --preprocessing_num_workers 1 \
        --drop_outliner_lengths \
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

run function_completion
