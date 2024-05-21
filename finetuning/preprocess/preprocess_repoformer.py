'''
This file reads JSONL files of raw code data and splits the entire dataset into
train/valid/test splits and stores as arrow datasets.

This script also chunks and tokenizes train, validation, test splits of the arrow
dataset. The information stored in the output dataset contains the chunked string,
location of original string, and information like chunk index to uniquely
identify each chunk.

'''
import argparse
import glob
import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

from sklearn.model_selection import train_test_split
import numpy as np
import random


args, tokenizer = None, None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
        help='Directory to the data directory containing JSONL files with raw data')
    parser.add_argument('--output_dir', type=str, 
        help='Directory to store preprocessed dataset')
    parser.add_argument('--tokenizer_name', type=str, 
        default='bigcode/starcoderbase-1b')
    parser.add_argument('--seed', type=int, default=0, 
        help='Seed value to use while shuffling in splitting')
    parser.add_argument('--test_and_valid_combined_size', type=float,
        default=0.01, help='% value of how much data to use for valid+test')
    parser.add_argument('--seq_length', type=int, default=2048,
        help='Maximum sequence length while tokenizing')
    parser.add_argument('--max_cfc_length', type=int, default=1024,
        help='Maximum length of cross-file context after tokenized')
    parser.add_argument('--lc_rc_ratio', type=float, default=2.0, 
        help='Left context budget / right context budget (ratio of seq_length after adding target lines, cross-file context lines, and special tokens)')
    parser.add_argument('--pos_label_es_gain_threshold', type=float, default=0.0,
        help='Gain threshold for utilizing cross-file context')
    parser.add_argument('--num_proc', type=int, default=96,
        help='Number of processes for parallel processing')
    parser.add_argument('--sanity_check_fim', action='store_true',
        help='Sanity check (FIM only, no CLM, no CFC)')
    parser.add_argument('--sanity_check_clm', action='store_true',
        help='Sanity check (CLM only, no FIM, no CFC)')
    parser.add_argument('--add_end_cfc_to_neg', action='store_true',
        help='Add </cfc_info> to negative cases')
    parser.add_argument('--cfc_in_rc', action='store_true',
        help='Use the cfc in rc prompting style. Place cfc to the left of <fim_middle> and use a <end_rc> token to separate with rc.')
    parser.add_argument('--add_neg_retrieval', action='store_true',
        help='Create RAG training cases with negative retrieval results. Will use a special token following <end_rc> to indicate this.')
    parser.add_argument('--neg_retrieval_es_decrease', type=float, default=0.1,
        help='Threshold of ES decrease for a case to be considered as negative retrieval.')
    
    return parser.parse_args()


def read_data(data_dir):
    '''Funtion to read the processed json datasets'''
    all_files = glob.glob(os.path.join(data_dir, '*.jsonl'))
    all_data = {'content': [], 'origin': [], 'origin_index': []}
    for filepath in tqdm(all_files, desc='reading data files'):
        origin = '/'.join(filepath.split('/')[-2:])
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                raw_str = json.loads(line)
                all_data['content'].append(raw_str)
                all_data['origin'].append(origin)
                all_data['origin_index'].append(line_num)
    return all_data


def dataset_from_all_data(args, all_data):
    '''Split `all_data` into train/valid/test splits'''
    
    all_repos = list(set(['/'.join(x['metadata']['task_id'].split('/')[:2]) for x in all_data['content']]))
    train_repos, testvalid_repos = train_test_split(all_repos, test_size=args.test_and_valid_combined_size, 
                                                    shuffle=True, random_state=args.seed)
    valid_repos, test_repos = train_test_split(testvalid_repos, test_size=0.5, shuffle=True, random_state=args.seed)
    train_repos = set(train_repos)
    valid_repos = set(valid_repos)
    test_repos = set(test_repos)

    train_data, valid_data, test_data = {k: [] for k in all_data.keys()}, {k: [] for k in all_data.keys()}, {k: [] for k in all_data.keys()}
    for i, entry in enumerate(all_data['content']):
        origin = '/'.join(entry['metadata']['task_id'].split('/')[:2])
        if origin in train_repos:
            for k in all_data.keys():
                train_data[k].append(all_data[k][i])
        elif origin in valid_repos:
            for k in all_data.keys():
                valid_data[k].append(all_data[k][i])
        else:
            for k in all_data.keys():
                test_data[k].append(all_data[k][i])

    split_dataset = DatasetDict({
        'train': Dataset.from_dict(train_data),
        'test': Dataset.from_dict(test_data),
        'valid': Dataset.from_dict(valid_data)
    })
    print(len(all_repos))
    print(len(train_repos), len(valid_repos), len(test_repos))
    print({
        'train': len(split_dataset['train']),
        'valid': len(split_dataset['valid']),
        'test': len(split_dataset['test'])
    })

    return split_dataset


def tokenize(entry):
    global tokenizer, args
    
    if '####CLM_DATA' in entry['content']['metadata']['task_id']:
        lc_token_ids = tokenizer(entry['content']['prompt'], 
                                 padding=False, truncation=False, add_special_tokens=False).input_ids
        rc_token_ids = tokenizer(entry['content']['prompt'], 
                                    padding=False, truncation=False, add_special_tokens=False).input_ids  # placeholder
        tgt_token_ids = tokenizer(entry['content']['prompt'], 
                                    padding=False, truncation=False, add_special_tokens=False).input_ids  # placeholder
        cfc_token_ids = tokenizer(entry['content']['prompt'], 
                                    padding=False, truncation=False, add_special_tokens=False).input_ids  # placeholder
        
        # just placeholder
        entry['content']['metadata']['es_infile'] = entry['content']['es_infile']
        entry['content']['metadata']['es_rg1'] = entry['content']['es_rg1']

        ret_dict = {
            'metadata': entry['content']['metadata'],
            'lc_token_ids': lc_token_ids,
            'tgt_token_ids': tgt_token_ids,
            'rc_token_ids': rc_token_ids,
            'cfc_token_ids': cfc_token_ids,
        }
    else:
        # tokenize
        lc_token_ids = tokenizer(entry['content']['prompt'], 
                                    padding=False, truncation=False, add_special_tokens=False).input_ids
        rc_token_ids = tokenizer(entry['content']['right_context'], 
                                    padding=False, truncation=False, add_special_tokens=False).input_ids
        tgt_token_ids = tokenizer(entry['content']['groundtruth'], 
                                    padding=False, truncation=False, add_special_tokens=False).input_ids
        cfc_token_ids = tokenizer(entry['content']['crossfile_context'], 
                                    padding=False, truncation=False, add_special_tokens=False).input_ids
        
        cfc_token_ids = cfc_token_ids[:args.max_cfc_length]

        # record scores before and after RAG
        entry['content']['metadata']['es_infile'] = entry['content']['es_infile']
        entry['content']['metadata']['es_rg1'] = entry['content']['es_rg1']

        ret_dict = {
            'metadata': entry['content']['metadata'],
            'lc_token_ids': lc_token_ids,
            'tgt_token_ids': tgt_token_ids,
            'rc_token_ids': rc_token_ids,
            'cfc_token_ids': cfc_token_ids,
        }
    return ret_dict


def concat_context(entry):
    global tokenizer, args

    if '####CLM_DATA' in entry['content']['metadata']['task_id']:
        # CLM cases
        token_ids = entry['tgt_token_ids'][:args.seq_length]
        if len(token_ids) < args.seq_length:
            pad = [tokenizer.pad_token_id] * (args.seq_length - len(token_ids))
            token_ids = token_ids + pad
        ret_dict = {
            'metadata': entry['metadata'],
            'token_ids': token_ids,
        }

    else:
        # with cfc label
        if args.sanity_check_clm or args.sanity_check_fim:
            use_cfc = False
            is_negative_retrieval = False
        else:
            use_cfc = entry['content']['metadata']['es_rg1'] - entry['content']['metadata']['es_infile'] > args.pos_label_es_gain_threshold
            is_negative_retrieval = entry['content']['metadata']['es_infile'] - entry['content']['metadata']['es_rg1'] > args.neg_retrieval_es_decrease
            assert not (is_negative_retrieval and use_cfc)

        # truncate lc and rc 
        # format 1 (use_cfc): <fim_prefix> lc <fim_suffix> rc <fim_middle> <cfc_info> cfc </cfc_info> tgt
        # format 2 (not use_cfc): <fim_prefix> lc <fim_suffix> rc <fim_middle> tgt
        cfc_len = len(entry['cfc_token_ids'])
        tgt_len = len(entry['tgt_token_ids'])
        if use_cfc:
            lr_budget = args.seq_length - tgt_len - cfc_len - 5
        else:
            # lr_budget = args.seq_length - tgt_len - 3
            # we reduce the lr_budget for both positive and negative cases to avoid model learning shortcuts
            lr_budget = args.seq_length - tgt_len - cfc_len - 5

        rc_budget = int(lr_budget / (args.lc_rc_ratio + 1))
        lc_budget = int(rc_budget * args.lc_rc_ratio)
        entry['lc_token_ids'] = entry['lc_token_ids'][-lc_budget:]
        entry['rc_token_ids'] = entry['rc_token_ids'][:rc_budget]
        
        # format
        if args.sanity_check_clm:
            token_ids = entry['lc_token_ids'] + entry['tgt_token_ids'] + entry['rc_token_ids']
        else:
            fim_prefix_id = tokenizer.vocab['<fim_prefix>']
            fim_suffix_id = tokenizer.vocab['<fim_suffix>']
            fim_middle_id = tokenizer.vocab['<fim_middle>']

            if args.sanity_check_fim:
                if args.add_neg_retrieval:
                    raise NotImplementedError
                token_ids = [fim_prefix_id] + entry['lc_token_ids'] + [fim_suffix_id] + entry['rc_token_ids'] + [fim_middle_id] + entry['tgt_token_ids']

            elif args.cfc_in_rc:
                cfc_info_start_id = tokenizer.vocab['<cfc_info>']
                end_right_context_id = tokenizer.vocab['<end_rc>']
                if args.add_neg_retrieval:
                    neg_cfc_info_start_id = tokenizer.vocab['<neg_cfc_info>']
                
                if use_cfc:
                    token_ids = [fim_prefix_id] + entry['lc_token_ids'] + [fim_suffix_id] + entry['rc_token_ids'] + [end_right_context_id] + [cfc_info_start_id] + entry['cfc_token_ids'] + [fim_middle_id] + entry['tgt_token_ids']
                elif args.add_neg_retrieval and is_negative_retrieval:
                    assert not args.add_end_cfc_to_neg
                    token_ids = [fim_prefix_id] + entry['lc_token_ids'] + [fim_suffix_id] + entry['rc_token_ids'] + [end_right_context_id] + [neg_cfc_info_start_id] + [fim_middle_id] + entry['tgt_token_ids']
                else:
                    assert not args.add_end_cfc_to_neg
                    token_ids = [fim_prefix_id] + entry['lc_token_ids'] + [fim_suffix_id] + entry['rc_token_ids'] + [end_right_context_id] + [fim_middle_id] + entry['tgt_token_ids']

            else:
                if args.add_neg_retrieval:
                    raise NotImplementedError
                cfc_info_start_id = tokenizer.vocab['<cfc_info>']
                cfc_info_end_id = tokenizer.vocab['</cfc_info>']
                if use_cfc:
                    token_ids = [fim_prefix_id] + entry['lc_token_ids'] + [fim_suffix_id] + entry['rc_token_ids'] + [fim_middle_id] + [cfc_info_start_id] + entry['cfc_token_ids'] + [cfc_info_end_id] + entry['tgt_token_ids']
                else:
                    if args.add_end_cfc_to_neg:
                        token_ids = [fim_prefix_id] + entry['lc_token_ids'] + [fim_suffix_id] + entry['rc_token_ids'] + [fim_middle_id] + [cfc_info_end_id] + entry['tgt_token_ids']
                    else:
                        token_ids = [fim_prefix_id] + entry['lc_token_ids'] + [fim_suffix_id] + entry['rc_token_ids'] + [fim_middle_id] + entry['tgt_token_ids']

        # pad to the left
        # if len(token_ids) < args.seq_length:
        #     pad = [tokenizer.pad_token_id] * (args.seq_length - len(token_ids))
        #     token_ids = pad + token_ids

        # we pad to the right to avoid the length shortcut 
        if len(token_ids) < args.seq_length:
            pad = [tokenizer.pad_token_id] * (args.seq_length - len(token_ids))
            token_ids = token_ids + pad

        ret_dict = {
            'metadata': entry['metadata'],
            'token_ids': token_ids,
        }
    return ret_dict


def main():
    global args, tokenizer
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.padding_side = 'right'
    if args.tokenizer_name == 'bigcode/starcoderbase-1b' and not args.sanity_check_fim and not args.sanity_check_clm:   
        if args.cfc_in_rc:
            if args.add_neg_retrieval:
                tokenizer.add_tokens(['<cfc_info>', '<end_rc>', '<neg_cfc_info>'])
                assert len(tokenizer.vocab) == 49155
            else:
                tokenizer.add_tokens(['<cfc_info>', '<end_rc>'])
                assert len(tokenizer.vocab) == 49154
        else: 
            tokenizer.add_tokens(['<cfc_info>', '</cfc_info>'])
            assert len(tokenizer.vocab) == 49154
    else:
        assert args.sanity_check_clm or args.sanity_check_fim

    if 'starcoder' in args.tokenizer_name.lower():
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'codegen' in args.tokenizer_name.lower():
        tokenizer.pad_token_id = 50256
    else:
        raise NotImplementedError
    print('Set tokenizer pad token id to', tokenizer.pad_token_id)

    # gather all data from desired data version
    all_data = read_data(args.data_dir)

    # split data
    dataset = dataset_from_all_data(args, all_data)

    # tokenize
    dataset = dataset.map(tokenize, num_proc=args.num_proc)  # batched=False

    # truncate and form the final text
    if args.sanity_check_fim:
        print('Warning: --sanity_check_fim specified. Will not create cases containing <cfc_info>.')
    elif args.sanity_check_clm:
        print('Warning: --sanity_check_clm specified. Will not create cases containing <cfc_info>.')
    dataset = dataset.map(concat_context, num_proc=args.num_proc)

    # filter out too long items
    dataset = dataset.filter(lambda example: len(example['token_ids']) <= args.seq_length)
    print('After filtering by max token length:')
    print({'train': len(dataset['train']), 'valid': len(dataset['valid']), 'test': len(dataset['test'])})

    # save data
    dataset.save_to_disk(args.output_dir)


if __name__ == '__main__':
    main()
    
