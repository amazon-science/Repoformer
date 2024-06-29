#!/usr/bin/env python
# coding=utf-8

import os
import json
import torch
import torch.nn.functional as F
import logging
import argparse
import numpy as np

import torch.multiprocessing as mp
import torch.distributed as dist
from collections import Counter

from tqdm import tqdm
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    default_data_collator,
    AutoTokenizer,
    set_seed,
    AutoModelForCausalLM
)

from eval_metric import compute_metric_stmt
from eval_metric_cceval import compute_metric_stmt_cceval
from datetime import datetime
import time

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


def custom_data_collator(features):
    first = features[0]
    batch = {}
    for k, v in first.items():
        if v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if v is not None and isinstance(v, str):
            batch[k] = [f[k] for f in features]

    return batch


def build_datasets(args, tokenizer):
    # Initialize the model and tokenizer
    # when generating, we will use the logits of right-most token to predict the next token
    # so the padding should be on the left
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token

    # load the files into Dataset
    raw_datasets = load_dataset("json", data_files=args.prompt_file, cache_dir=args.cache_dir)
    raw_datasets = raw_datasets["train"]
    raw_datasets = raw_datasets.map(lambda example, idx: {'index': idx, **example}, with_indices=True)
    index2taskid = {idx: md["task_id"] for idx, md in zip(raw_datasets["index"], raw_datasets["metadata"])}
    column_names = raw_datasets.column_names

    def prepare_features(examples):
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            examples["prompt"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length - args.gen_length
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    def prepare_features_fim(examples):
        # first do proper truncation 
        tokenizer.truncation_side = "left"
        tokenized_inputs = tokenizer(
            examples["prompt"],
            padding=False,
            max_length=args.max_seq_length - args.gen_length - 10,
            truncation=True,
        )
        # inject fim tokens and redo tokenization
        input_text = ['<fim_prefix>' + x + '<fim_suffix>' + '<fim_middle>' for x in tokenizer.batch_decode(tokenized_inputs['input_ids'])]
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            # truncation=True,
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    def prepare_features_cfc_fim(examples):
        in_file_seq_length = args.max_seq_length - args.right_context_length - args.gen_length

        tokenizer.truncation_side = "right"
        cfc_features = tokenizer(
            examples["crossfile_context"] if type(examples["crossfile_context"]) == str else [x['text'] for x in examples["crossfile_context"]],
            padding=False,
            truncation=True,
            max_length=args.cfc_seq_length - 5
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length - 5
        )

        input_text = ['<fim_prefix>' + y + '<fim_suffix>' + x + '<fim_middle>' for x, y in zip(tokenizer.batch_decode(cfc_features['input_ids']),
                                                                                               tokenizer.batch_decode(infile_seq_features['input_ids']))]
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            truncation=False
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    def prepare_features_cfc(examples):
        in_file_seq_length = args.max_seq_length - args.cfc_seq_length - args.gen_length

        tokenizer.truncation_side = "right"
        crossfile_seq_features = tokenizer(
            examples["crossfile_context"] if type(examples["crossfile_context"][0]) == str else [x['text'] for x in examples["crossfile_context"]],
            truncation=True,
            max_length=args.cfc_seq_length
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length
        )

        # concatenate project-level context and file-level context
        features = {}
        for k, v in infile_seq_features.items():
            features[k] = []
            for idx, e in enumerate(v):
                iids = crossfile_seq_features[k][idx] + e
                features[k].append(iids)

        # pad to max_seq_length
        tokenizer.padding_side = "left"
        features = tokenizer.pad(features, padding="max_length", max_length=args.max_seq_length - args.gen_length)
        features["index"] = examples["index"]
        return features
    
    def prepare_features_leftright_context_fim(examples):
        in_file_seq_length = args.max_seq_length - args.right_context_length - args.gen_length

        tokenizer.truncation_side = "right"
        right_context_features = tokenizer(
            examples["right_context"],
            padding=False,
            truncation=True,
            max_length=args.right_context_length - 10
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length - 10
        )

        input_text = ['<fim_prefix>' + y + '<fim_suffix>' + x + '<fim_middle>' for x, y in zip(tokenizer.batch_decode(right_context_features['input_ids']),
                                                                                               tokenizer.batch_decode(infile_seq_features['input_ids']))]
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            truncation=False
        )

        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    def prepare_features_leftright_context(examples):
        in_file_seq_length = args.max_seq_length - args.right_context_length - args.gen_length

        tokenizer.truncation_side = "right"
        right_context_features = tokenizer(
            examples["right_context"],
            truncation=True,
            max_length=args.cfc_seq_length
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length
        )

        # concatenate project-level context and file-level context
        features = {}
        for k, v in infile_seq_features.items():
            features[k] = []
            for idx, e in enumerate(v):
                iids = right_context_features[k][idx] + e
                features[k].append(iids)

        # pad to max_seq_length
        tokenizer.padding_side = "left"
        features = tokenizer.pad(features, padding="max_length", max_length=args.max_seq_length - args.gen_length)
        features["index"] = examples["index"]
        return features
    
    def prepare_features_right_cfc_left(examples):
        in_file_seq_length = args.max_seq_length - args.cfc_seq_length - args.right_context_length - args.gen_length

        tokenizer.truncation_side = "right"
        right_context_features = tokenizer(
            examples["right_context"],
            truncation=True,
            max_length=args.right_context_length
        )
        crossfile_seq_features = tokenizer(
            examples["crossfile_context"] if type(examples["crossfile_context"][0]) == str else [x['text'] for x in examples["crossfile_context"]],
            truncation=True,
            max_length=args.cfc_seq_length
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length
        )

        # concatenate project-level context and file-level context
        features = {}
        for k, v in infile_seq_features.items():
            features[k] = []
            for idx, e in enumerate(v):
                iids = right_context_features[k][idx] + crossfile_seq_features[k][idx] + e
                features[k].append(iids)

        # pad to max_seq_length
        tokenizer.padding_side = "left"
        features = tokenizer.pad(features, padding="max_length", max_length=args.max_seq_length - args.gen_length)
        features["index"] = examples["index"]
        return features

    def prepare_features_right_cfc_left_fim(examples):
        in_file_seq_length = args.max_seq_length - args.cfc_seq_length - args.right_context_length - args.gen_length

        tokenizer.truncation_side = "right"
        cfc_features = tokenizer(
            examples["crossfile_context"] if type(examples["crossfile_context"][0]) == str else [x['text'] for x in examples["crossfile_context"]],
            padding=False,
            truncation=True,
            max_length=args.cfc_seq_length - 10
        )
        right_context_features = tokenizer(
            examples["right_context"],
            padding=False,
            truncation=True,
            max_length=args.right_context_length - 10
        )
        tokenizer.truncation_side = "left"
        infile_seq_features = tokenizer(
            examples["prompt"],
            truncation=True,
            max_length=in_file_seq_length - 10
        )

        input_text = ['<fim_prefix>' + x + z + '<fim_suffix>' + y + '<fim_middle>' for x, y, z in zip(tokenizer.batch_decode(cfc_features['input_ids']), 
                                                                                                      tokenizer.batch_decode(right_context_features['input_ids']),
                                                                                                      tokenizer.batch_decode(infile_seq_features['input_ids']))]
        tokenizer.padding_side = "left"
        tokenized_inputs = tokenizer(
            input_text,
            padding="max_length",
            max_length=args.max_seq_length - args.gen_length,
            truncation=False
        )
        features = {k: t for k, t in tokenized_inputs.items()}
        features["index"] = examples["index"]
        return features
    
    if args.model_type == "codelm":
        if args.use_fim_prompt:
            if 'starcoder' not in args.model_name_or_path.lower():
                print('Warning: unrecognized model name, starcoder prompt is used as default.')
            prep_function = prepare_features_fim
            tokenized_datasets = raw_datasets.map(
                prep_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset (FIM mode set to true)",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                prepare_features,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    elif args.model_type == "codelm_cfc":
        if args.use_fim_prompt:
            if 'starcoder' not in args.model_name_or_path.lower():
                print('Warning: unrecognized model name, starcoder prompt is used as default.')
            prep_function = prepare_features_cfc_fim
            tokenized_datasets = raw_datasets.map(
                prep_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                prepare_features_cfc,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    elif args.model_type == "codelm_leftright_context":
        if args.use_fim_prompt:
            if 'starcoder' not in args.model_name_or_path.lower():
                print('Warning: unrecognized model name, starcoder prompt is used as default.')
            prep_function = prepare_features_leftright_context_fim
            tokenized_datasets = raw_datasets.map(
                prep_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                prepare_features_leftright_context,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    elif args.model_type == "codelm_right_cfc_left":
        if args.use_fim_prompt:
            if 'starcoder' not in args.model_name_or_path.lower():
                print('Warning: unrecognized model name, starcoder prompt is used as default.')
            prep_function = prepare_features_right_cfc_left_fim
            tokenized_datasets = raw_datasets.map(
                prep_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                prepare_features_right_cfc_left,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
    else:
        raise NotImplementedError("prepare feature functions not implemented for new model type")

    if args.drop_outliner_lengths:
        c = Counter([len(x['input_ids']) for x in tokenized_datasets])
        print('Input length distribution:', c)
        if len(c) > 1:
            tokenized_datasets_new, index2taskid_new = [], {}
            added = 0
            for i, entry in enumerate(tokenized_datasets):
                if len(entry['input_ids']) > c.most_common(1)[0][0]:
                    continue
                else:
                    tokenized_datasets_new.append(entry)
                    entry["index"] = added
                    index2taskid_new[added] = index2taskid[i]
                    added += 1
            print('Droping input with length larger than {}, {} remaining'.format(c.most_common(1)[0][0], len(tokenized_datasets_new)))
        else:
            print('No outliners found. Ignoring the --drop_outliner_lengths flag.')
            tokenized_datasets_new, index2taskid_new = tokenized_datasets, index2taskid
        return tokenized_datasets_new, index2taskid_new
    else:
        return tokenized_datasets, index2taskid


def model_inference(tokenized_datasets, index2taskid, tokenizer):
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    elif args.dtype == 'int8':
        dtype = torch.int8
    else:
        assert False, f'{args.dtype=} not implemented'

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=dtype, 
                                                 trust_remote_code=True, 
                                                 load_in_8bit=True if dtype == torch.int8 else False)
    
    # set up speculative decoding
    if args.draft_model:
        draft_model = AutoModelForCausalLM.from_pretrained(
            args.draft_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            load_in_8bit=True if dtype == torch.int8 else False
        )
        # repoformer
        if 'repoformer' in args.draft_model.lower():
            draft_model.resize_token_embeddings(len(tokenizer))
        print('Loaded {} as the draft model for speculative decoding.'.format(args.draft_model))

    logger.info(f'{model.dtype=}') 
    logger.info(args)

    total_samples_cnt = len(tokenized_datasets)
    logger.info(f"total samples: {total_samples_cnt}")

    data_sampler = SequentialSampler(tokenized_datasets)
    dataloader = DataLoader(
        tokenized_datasets,
        sampler=data_sampler,
        collate_fn=custom_data_collator,
        batch_size=args.batch_size
    )

    # model = accelerator.prepare_model(model)
    accelerator = Accelerator()
    model = model.to(accelerator.device)
    if args.draft_model:
        draft_model = draft_model.to(accelerator.device)
    dataloader = accelerator.prepare(dataloader)

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
    all_preds = []
    all_task_ids = []
    all_batch_latency, all_batch_latency_per_token = [], []
    all_batch_max_prob_first_token, all_batch_max_prob_lower_bound, all_batch_max_prob_all_tokens, all_batch_entropy_first_token, all_batch_entropy_all_tokens = [], [], [], [], []

    model.eval()
    if args.draft_model:
        draft_model.eval()
        
    # warm up
    if args.log_latency:
        print('warming up...')
        tokenized_datasets_warmup = tokenized_datasets.select(range(60))
        data_sampler_warmup = SequentialSampler(tokenized_datasets_warmup)
        dataloader_warmup = DataLoader(
            tokenized_datasets_warmup,
            sampler=data_sampler_warmup,
            collate_fn=custom_data_collator,
            batch_size=args.batch_size
        )
        dataloader_warmup = accelerator.prepare(dataloader_warmup)
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(dataloader_warmup), total=len(dataloader_warmup)):
                output_sequences = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=args.max_seq_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                )
        
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            cur_batch_start_time = time.time_ns()
            if args.draft_model:
                output_sequences = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=args.max_seq_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    assistant_model=draft_model,
                    num_assistant_tokens=args.lookahead, 
                    num_assistant_tokens_schedule=args.lookahead_strategy,
                )
            else:
                output_sequences = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    max_length=args.max_seq_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    pad_token_id=tokenizer.pad_token_id,
                    output_scores=args.log_uncertainty,
                    return_dict_in_generate=args.log_uncertainty
                )
                if args.log_uncertainty:
                    # uncertainty calculation
                    logits = torch.concatenate(output_sequences.scores, dim=0)
                    probabilities = F.softmax(logits, dim=-1)
                    max_probs, _ = torch.max(probabilities, dim=-1)
                    log_probs = torch.log(probabilities)
                    entropy = -torch.sum(probabilities * log_probs, dim=-1)
                    output_sequences = output_sequences.sequences
                    
                    # metrics
                    max_prob_first_token = max_probs[0]
                    max_prob_lower_bound = max_probs.min()
                    max_prob_all_tokens = max_probs.mean()
                    entropy_first_token = entropy[0]
                    entropy_all_tokens = entropy.mean()
                    
            cur_batch_end_time = time.time_ns()
            cur_batch_latency = torch.tensor((cur_batch_end_time - cur_batch_start_time) / 1_000_000_000).to(batch['index'].device)
            
            batch_task_id = batch["index"]
            batch_pred = accelerator.pad_across_processes(output_sequences, dim=1, pad_index=tokenizer.pad_token_id)
            batch_task_id, batch_pred, cur_batch_latency = accelerator.gather((batch_task_id, batch_pred, cur_batch_latency))
            if args.log_uncertainty:
                max_prob_first_token, max_prob_lower_bound, max_prob_all_tokens, entropy_first_token, entropy_all_tokens = accelerator.gather((max_prob_first_token, max_prob_lower_bound, max_prob_all_tokens, entropy_first_token, entropy_all_tokens))
                all_batch_max_prob_first_token.extend([x.item() for x in max_prob_first_token])
                all_batch_max_prob_lower_bound.extend([x.item() for x in max_prob_lower_bound])
                all_batch_max_prob_all_tokens.extend([x.item() for x in max_prob_all_tokens])
                all_batch_entropy_first_token.extend([x.item() for x in entropy_first_token])
                all_batch_entropy_all_tokens.extend([x.item() for x in entropy_all_tokens])

            batch_pred = batch_pred[:, -args.gen_length:]
            generated_texts = tokenizer.batch_decode(batch_pred, skip_special_tokens=False)
            if 'starcoder2' in args.model_name_or_path:
                generated_texts = [x.split('<file_sep>')[0] for x in generated_texts]
            all_preds.extend(generated_texts)
            all_task_ids.extend(batch_task_id.tolist())
            all_batch_latency.extend(cur_batch_latency.tolist() if type(cur_batch_latency.tolist()) != float else [cur_batch_latency.tolist()])
            all_batch_latency_per_token.extend((cur_batch_latency/args.gen_length).tolist() if type(cur_batch_latency.tolist()) != float else [(cur_batch_latency/args.gen_length).tolist()])
    print('Average latency per batch is {}s. Average latency per token is {}s'.format(np.mean(all_batch_latency), np.mean(all_batch_latency_per_token)))

    with open(f"{args.output_dir}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
        id_processed = set()
        for idx, p in zip(all_task_ids, all_preds):
            if index2taskid[idx] not in id_processed:
                out_dict = {"task_id": index2taskid[idx], "pred": p}
                if args.log_latency:
                    out_dict["sample_latency"] = all_batch_latency[idx]
                    out_dict["sample_latency_per_token"] = all_batch_latency_per_token[idx]
                if args.log_uncertainty:
                    out_dict["max_prob_first_token"] = all_batch_max_prob_first_token[idx]
                    out_dict["max_prob_lower_bound"] = all_batch_max_prob_lower_bound[idx]
                    out_dict["max_prob_all_tokens"] = all_batch_max_prob_all_tokens[idx]
                    out_dict["entropy_first_token"] = all_batch_entropy_first_token[idx]
                    out_dict["entropy_all_tokens"] = all_batch_entropy_all_tokens[idx]

                f_pred.write(json.dumps(out_dict) + "\n")
                id_processed.add(index2taskid[idx])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model inference args
    parser.add_argument("--language", type=str, required=True, help="language name")
    parser.add_argument("--model_name_or_path", default=None, type=str, help="Pre-trained Model Path")
    parser.add_argument("--tokenizer_name", default=None, type=str)
    parser.add_argument(
        "--model_type",
        type=str,
        default="codelm",
        choices=["codelm", "codelm_cfc", "codelm_leftright_context", 'codelm_right_cfc_left'],
        help="Model type to be loaded"
    )
    parser.add_argument("--use_fim_prompt", action='store_true', help="Use FIM prompting style (StarCoder, CodeGen-2, SantaCoder, etc.)")
    parser.add_argument("--prompt_file", type=str, default=None, help="file with a list of prompts")
    parser.add_argument("--gen_length", type=int, default=50, help="max length of generated token sequence")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="max length of prompt")
    parser.add_argument(
        "--cfc_seq_length",
        type=int,
        default=512,
        help="For model_type=codelm_cfc: Text sequence length corresponding to the retrieved nodes"
    )
    parser.add_argument(
        "--right_context_length",
        type=int,
        default=512,
        help="For model_type=codelm_leftright_context: Text sequence length corresponding to the right context"
    )
    parser.add_argument(
        "--min_cfc_score",
        type=float,
        default=0,
        help="For model_type=codelm_cfc: Text sequence length corresponding to the retrieved nodes"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for code completion")
    parser.add_argument("--stop_token", type=str, default=None, help="Token at which text generation is stopped")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling"
    )
    parser.add_argument("--output_dir", type=str, default="output_dir", help="output directory to save predictions")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="The parameter for repetition penalty.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing."
    )
    parser.add_argument(
        "--overwrite_cache",
        type=bool,
        default=False,
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--dtype", type=str, default='fp16')
    parser.add_argument("--do_sample", action="store_true", help="whether we do sampling or greedy/beam-search")
    parser.add_argument("--num_beams", type=int, default=1, help="num of beam for beam-search")
    # hack to drop too long items
    parser.add_argument("--drop_outliner_lengths", action='store_true', help="drop outliers after preprocessing")

    # compute metric args
    parser.add_argument(
        "--ts_lib",
        type=str,
        default="build/python-lang-parser.so",
        help="tree-sitter lib for tokenize code"
    )
    parser.add_argument("--only_compute_metric", action="store_true", help="only compute metric")
    parser.add_argument(
        "--task",
        choices=["line_completion", "api_completion", "function_completion"],
        default="line_completion",
        help="task name"
    )
    
    # for speculative decoding
    parser.add_argument("--draft_model", type=str, default=None, help="draft model for speculative decoding")
    parser.add_argument("--lookahead", type=int, default=None, help="lookahead for speculative decoding")
    parser.add_argument("--lookahead_strategy", type=str, default=None, choices=['heuristic', 'constant'],
                        help="strategy for setting the lookahead for speculative decoding")

    # for cceval metric
    parser.add_argument("--compute_cceval_metric", action='store_true', help="use cceval metric")
    
    # logging latency
    parser.add_argument("--log_latency", action='store_true', help="log latency in the results file")

    # log uncertainty
    parser.add_argument("--log_uncertainty", action='store_true', help="log uncertainty in the results file")
    
    args = parser.parse_args()

    accelerator = Accelerator()
    if not args.only_compute_metric:
        if args.tokenizer_name is not None:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
        else:
            if 'no_fim' in args.model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path.strip('no_fim'), trust_remote_code=True)
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        tokenized_datasets, index2taskid = build_datasets(args, tokenizer)
        model_inference(tokenized_datasets, index2taskid, tokenizer)

    if args.log_uncertainty or args.log_latency:
        assert args.batch_size == 1
        
    # check if the process is the main process
    if accelerator.is_main_process:
        if args.compute_cceval_metric:
            compute_metric_stmt_cceval(args)
        else:
            compute_metric_stmt(args)
