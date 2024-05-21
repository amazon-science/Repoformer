import argparse
import os

import torch


def add_program_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expt_prefix", type=str, help="can use the training data name as experment name", 
        default="BigQuery")
    parser.add_argument("--train_datadir", type=str, help="path to the processed [PyPI + BigQuery] or [Wikitext_103] .arrow dataset")
    parser.add_argument("--valid_datadir", type=str, help="path to the processed [PyPI + BigQuery] or [Wikitext_103] .arrow dataset")
    parser.add_argument("--log_dir", type=str, default="../results/", help="Path of the Tensorboard log directory")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=42, help="value to seed RNG of torch, numpy")
    parser.add_argument("--track_steps", action="store_true", help="if True, progress bar will track training batches, else will track epochs")
    parser.add_argument("--save_step_frequency", default=1000, help="Number of steps (update steps) between saving checkpoints", type=int)
    return parser

def add_pl_args(parent_parser):
    parser = parent_parser.add_argument_group("pl.Trainer")
    parser.add_argument("--val_check_interval", type=int, 
        help="Validation frequency (specify interval in # of training steps, not batches)", default=1000)
    parser.add_argument("--devices", type=int, help="Number of gpu/cpu cores to use", default=8)
    parser.add_argument("--accelerator", type=str, help="Number of gpu/cpu cores to use", default="gpu")
    parser.add_argument("--log_every_n_steps", type=int, help="Logging frequency (in update steps)", default=100)
    parser.add_argument("--accumulate_grad_batches", type=int, help="Gradient accumulation steps", default=1) 
    parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--num_training_examples", type=int, default=-1, help="Number of training examples")
    parser.add_argument("--max_steps", type=int, default=-1, help="Number of training steps")
    parser.add_argument("--max_epochs", type=int, default=-1, help="Number of training epochs")
    parser.add_argument("--default_root_dir", type=str, required=True, help="Root dir")
    parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--debug_cuda_mem", action="store_true", help="Print GPU util")
    parser.add_argument("--precision", type=int, default=32, help="training precision")
    parser.add_argument("--ds_config", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deepspeed', 'stage2.json'), help="deepspeed config")
    return parent_parser

def add_model_args(parent_parser):
    parser = parent_parser.add_argument_group("LitCodeGen")
    # CodeGen specific arguments
    parser.add_argument("--model_name", type=str, default='gpt2') #, choices=["gpt2", "gpt2-large", "Salesforce/codegen-350M-mono", "bigcode/starcoderbase-1b"])
    parser.add_argument("--pad_token_id", type=int, default=50256)  # see here https://github.com/salesforce/CodeGen/blob/2ca076874ca2d26c2437df2968f6c43df92748bc/jaxformer/hf/sample.py#L201
    parser.add_argument("--dropout_layers", type=int, default=-1, help="Number of layers to add dropout to; if -1, dropout will be added to all layers; if 0, no dropout will be used")
    parser.add_argument("--dropout_p", type=float, default=0.1, help="Value of dropout probability to be added")
    parser.add_argument("--functional_dropout", action="store_true", help="If True, will use functional dropout on the token level representations")
    # training
    parser.add_argument("--no_scheduling", action="store_true", help="If True, will not use linear warmup with scheduling")
    parser.add_argument("--inv_sqrt_scheduling", action="store_true", help="If True, will use inverse square root schedule as in PICL.")
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0., help="L2 regularization")
    # dataloading
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size for training combined across all devices.")
    parser.add_argument("--valid_batch_size", type=int, default=64, help="Batch size for validation combined across all devices.")
    # objective
    parser.add_argument("--loss", type=str, help="Loss function name", default="MLE_Only", choices=["MLE_Only",  "Repoformer"])
    # args for repoformer    
    parser.add_argument("--full_sequence_code_completion_loss", action='store_true')
    parser.add_argument("--separate_cfc_token_loss", action='store_true')
    parser.add_argument("--cfc_token_loss_lambda", type=float, default=None)
    parser.add_argument("--replace_cfc_end_with_fim_middle", action='store_true')
    parser.add_argument("--cfc_in_rc", action='store_true')
    parser.add_argument("--has_neg_retrieval", action='store_true')
    parser.add_argument("--valid_with_cfc_f1", action='store_true')
    parser.add_argument("--debug_disable_adding_new_token", action='store_true')
    return parent_parser

def check_args(args):
    # sanity check on devices
    assert args.num_workers <= os.cpu_count(), "Number of dataloader workers cannot be greater than number of CPUs"
    if args.accelerator == "gpu":
        assert torch.cuda.is_available() and args.devices <= torch.cuda.device_count(), "Asking for more GPUs than available"
    elif args.accelerator == "cpu":
        assert args.devices <= os.cpu_count(), "Asking for more CPUs than available"
    # repoformer sanity checks
    if args.cfc_in_rc:
        assert not args.replace_cfc_end_with_fim_middle
        assert not args.debug_disable_adding_new_token
        assert args.separate_cfc_token_loss

