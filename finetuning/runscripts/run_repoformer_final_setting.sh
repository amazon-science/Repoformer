#!/usr/bin/env bash

export CUDA_HOME=/usr/local/cuda
export PYTHONPATH=${PYTHONPATH}:${HOME_DIR}/finetuning/

export HOME_DIR=`realpath ../../`

LR=2e-5
GRAD_STEPS=4
TRAIN_BS=128
VALID_BS=128
EPOCHS=2
STEPS=-1
WARMUP_STEPS=50
GPUS=0,1,2,3,4,5,6,7
GPU_COUNT=8
NODE_COUNT=1
NUM_WORKERS=96

LAMBDA_CFC=1.0

DATA_DIR="your_data_path"
TRAIN_DIR="${DATA_DIR}/20230908_diverse_chunk240k_func120k_starcoder1b_maxlen2048_lrratio2.0_cfcinrc/train"
VALID_DIR="${DATA_DIR}/20230908_diverse_chunk240k_func120k_starcoder1b_maxlen2048_lrratio2.0_cfcinrc/valid"
EXP_PREFIX="repoformer_1b_final_cfcinrc_lambdacfc${LAMBDA_CFC}"


CUDA_VISIBLE_DEVICES=$GPUS python pl_trainer.py \
   --num_workers $NUM_WORKERS \
   --devices $GPU_COUNT \
   --accelerator gpu \
   --model_name bigcode/starcoderbase-1b \
   --pad_token_id 0 \
   --dropout_p 0.1 \
   --expt_prefix ${EXP_PREFIX} \
   --default_root_dir ./logs_store/deepspeed/ \
   --train_datadir $TRAIN_DIR \
   --valid_datadir $VALID_DIR \
   --log_dir ./logs/ \
   --seed 1234 \
   --lr $LR \
   --weight_decay 0.1 \
   --gradient_clip_val 1.0 \
   --max_steps $STEPS \
   --max_epochs $EPOCHS \
   --warmup_steps $WARMUP_STEPS \
   --train_batch_size $TRAIN_BS \
   --valid_batch_size $VALID_BS \
   --accumulate_grad_batches $GRAD_STEPS \
   --log_every_n_steps 20 \
   --save_step_frequency 20 \
   --val_check_interval 20 \
   --debug_cuda_mem \
   --use_deepspeed \
   --ds_config ${HOME_DIR}/finetuning/deepspeed/stage3_1_bf16.json \
   --precision 16  \
   --loss Repoformer \
   --valid_with_cfc_f1 \
   --separate_cfc_token_loss \
   --cfc_token_loss_lambda ${LAMBDA_CFC} \
   --cfc_in_rc 
