# Sample chunks and and corresponding cross-file contexts
import os
import re
import lzma
import json
import glob
import argparse
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import code_tokenize as ctok
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from functools import partial
import random


def tokenize_nltk(text):
    words = word_tokenize(text)
    output_list = []
    for w in words:
        w_list = re.findall(r'\w+', w)
        output_list.extend(w_list)
    return output_list


def preprocess_code_tokenizer(text, lang='python'):
    return ' '.join([x.text for x in ctok.tokenize(text, lang=lang, syntax_error='ignore')])


def get_preprocess_code_tokenizer(lang):
    return lambda x: preprocess_code_tokenizer(x, lang=lang)


def jaccard_similarity(tokenized_query, tokenized_doc, containment=False):
    set1 = set(tokenized_query)
    set2 = set(tokenized_doc)
    intersection = len(set1.intersection(set2))
    union = len(set1) if containment else len(set1.union(set2))
    return float(intersection) / union


def compute_jaccard_sim_x_to_list_y(x_entry, y_entries):
    out = []
    for y_entry in y_entries:
        out.append(jaccard_similarity(x_entry['text'], y_entry['text']))
    return out


def read_code(fname):
    with open(fname, 'r', encoding='utf8') as f:
        return f.read()
    

def iterate_repository(base_dir, repo):
    """
    Gather all .py files under basedir/repo/
    """
    pattern = os.path.join(f'{base_dir}/{repo}', "**", "*.py")
    files = glob.glob(pattern, recursive=True)

    skipped_files = []
    loaded_code_files = dict()
    base_dir_list = os.path.normpath(base_dir).split(os.sep)
    for fname in files:
        try:
            code = read_code(fname)
            fpath_tuple = tuple(os.path.normpath(fname).split(os.sep)[len(base_dir_list):])
            loaded_code_files[fpath_tuple]= code
        except Exception as e:
            skipped_files.append((fname, e))
            continue

    if len(skipped_files) > 0:
        print(f"Skipped {len(skipped_files)} out of {len(files)} files due to I/O errors")
        for fname, e in skipped_files:
            print(f"{fname}: {e}")

    return loaded_code_files


def file_to_chunks(file, args, window_stride_configs):
    """
    Extract chunks from a single file
    """
    # filenamefull = args.input_dir + '/' + '/'.join(file)
    # code = read_code(filenamefull)
    code = file['content']
    code_lines = [x for x in code.split('\n') if x.strip()]

    out_data = {}
    for k in window_stride_configs:
        window, stride = k.split('_')
        window = int(window)
        stride = int(stride)
        chunks = []
        for i in range(0, len(code_lines), stride):
            c = "\n".join(code_lines[i:i + window])
            tokenized_c = tokenize_nltk(c)
            if len(tokenized_c) > 0:
                chunks.append({
                    'filename': file['filepath'],
                    'start_i': i,
                    'end_i': i + window,
                    'text': c,
                    'text_tokenized': tokenized_c
                })
        out_data[k] = chunks

    return out_data, file['filepath'], code_lines


def get_cfc(entry, args, filename2codelines, all_cfc_dict, window_stride_configs):
    cur_key = None
    for key in window_stride_configs[1:]:
        span = 11 if args.oracle_in_query else 10
        if str((entry['end_i'] - entry['start_i']) * span) == key.split('_')[0]:
            cur_key = key
            break
    if cur_key is None:
        return {}
    
    lc_lines = filename2codelines[entry['filename']][:entry['start_i']]
    rc_lines = filename2codelines[entry['filename']][entry['end_i']:]
    curtext_lines = entry['text'].split('\n')
    curtext_nlines = len(curtext_lines)
    
    if args.oracle_in_query:
        query = '\n'.join(lc_lines[-5*curtext_nlines:] + curtext_lines + rc_lines[:5*curtext_nlines])
    else:
        query = '\n'.join(lc_lines[-5*curtext_nlines:] + rc_lines[:5*curtext_nlines])
    cfcs = [x for x in all_cfc_dict[cur_key] if x['filename'] != entry['filename']]

    tokenized_query = tokenize_nltk(query)
    tokenized_docs = [x['text_tokenized'] for x in cfcs]    # [tokenize_nltk(d) for d in cfcs]
    scores = [jaccard_similarity(tokenized_query, d, containment=False) for d in tokenized_docs]

    cfcs = [x for _, x in sorted(zip(scores, cfcs), key=lambda x: x[0], reverse=True)]
    topk_cfcs = cfcs[:args.topk_cfc]

    # multilingual-check
    if args.lang == 'python':
        line_start_sym = "#"
    elif args.lang in ['csharp', 'java', 'c', 'typescript', 'javascript']:
        line_start_sym = "//"
    else:
        raise NotImplementedError
    cfc_text = f"{line_start_sym} Here are some relevant code fragments from other files of the repo:\n\n"

    # multilingual-check
    for cur_cfc in topk_cfcs:
        cfc_text += f"{line_start_sym} the below code fragment can be found in:\n{line_start_sym} {cur_cfc['filename']}" + "\n"
        cfc_text += "\n".join([f"{line_start_sym} {cl}" for cl in cur_cfc['text'].strip('\n').splitlines()]) + "\n\n"

    ret_entry = {
        'metadata': {
            'task_id': f"{entry['filename']}/{entry['start_i']}_{entry['end_i']}",
            'filename': entry['filename'],
            'start_i': entry['start_i'],
            'end_i': entry['end_i'],
        },
        'prompt': '\n'.join(lc_lines) + '\n' if len(lc_lines) > 0 else "",
        'groundtruth': entry['text'],
        'right_context': '\n' + '\n'.join(rc_lines) if len(rc_lines) > 0 else "",
        'crossfile_context': cfc_text
    }

    # Strip the preceding whitespace of groundtruth and add to the end of prompt
    # indent_len = len(ret_entry['groundtruth']) - len(ret_entry['groundtruth'].lstrip())
    # prefix = ret_entry['groundtruth'][:indent_len]
    # ret_entry['groundtruth'] = ret_entry['groundtruth'][indent_len:]
    # ret_entry['prompt'] = ret_entry['prompt'] + prefix

    return ret_entry


def sample_chunks_with_cfc(args, repo_data, repo_name):
    """
    Sample chunks from a repository
    Return a list of candidate targets paired with top-k cfc
    """
    pool = mp.Pool(args.num_processes)

    # files = iterate_repository(repo_dir, repo_name)
    files = repo_data['files']
    for i in range(len(files)):
        files[i]['filepath'] = repo_name + '/' + files[i]['filepath']

    # step 1: sample chunks for both candidate and cfc
    window_stride_configs = ['10_10'] # , '10_5']
    for n_line in range(1, 10):
        if args.oracle_in_query:
            window = n_line * 11
        else:
            window = n_line * 10
        stride = window // 2
        window_stride_configs.append(f'{window}_{stride}')

    filename2codelines = {}
    candidate_targets = []
    all_cfc_dict = {k: [] for k in window_stride_configs}
    worker = partial(file_to_chunks, args=args, window_stride_configs=window_stride_configs)
    with tqdm(total=len(files), desc=f'{repo_name}: collecting chunks') as pbar:
        for chunk_dict, filename, codelines in pool.imap_unordered(worker, files):
            candidate_targets += chunk_dict[window_stride_configs[0]]
            for k in window_stride_configs:
                all_cfc_dict[k] += chunk_dict[k]
            filename2codelines[filename] = codelines
            pbar.update()

    # step 2: cluster
    if args.maximum_chunk_to_cluster < len(candidate_targets):
        candidate_targets = random.sample(candidate_targets, args.maximum_chunk_to_cluster)
    vectorizer = TfidfVectorizer(
        max_df=1.0,
        min_df=5,
        stop_words=None,
        tokenizer=(lambda x: x.split()),
        preprocessor=get_preprocess_code_tokenizer(args.lang),
    )
    candidate_targets_tfidf = vectorizer.fit_transform([x['text'] for x in candidate_targets])
    kmeans = KMeans(
        n_clusters=max(1, int(args.cluster_ratio*len(candidate_targets))),
        random_state=42
    ).fit(candidate_targets_tfidf)
    # cluster_ids, cluster_sizes = np.unique(kmeans.labels_, return_counts=True)
    # print(f"Number of elements assigned to each cluster: {cluster_sizes}")
    clusters = {}
    for candidate_target, cluster_id in zip(candidate_targets, kmeans.labels_):
        cluster_id = cluster_id.item()
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(candidate_target)

    # step 3: sample
    sampled_targets = []
    for cluster_id, cluster_targets in clusters.items():
        sampled_chunk = random.choice(cluster_targets)
        max_window_len = sampled_chunk['end_i'] - sampled_chunk['start_i']
        window_len = np.random.poisson(lam=args.poisson_lambda)
        window_len = max(1, window_len)
        window_len = min(window_len, max_window_len)
        sampled_startidx = random.randint(0, max_window_len - window_len)
        sampled_chunk = {
            'filename': sampled_chunk['filename'],
            'start_i': sampled_chunk['start_i'] + sampled_startidx,
            'end_i': sampled_chunk['start_i'] + sampled_startidx + window_len,
            'text': '\n'.join(sampled_chunk['text'].split('\n')[sampled_startidx:sampled_startidx + window_len])
        }

        # sanity check
        try:
            assert len(sampled_chunk['text'].strip()) > 0
            assert len(sampled_chunk['text'].split('\n')) == sampled_chunk['end_i'] - sampled_chunk['start_i']
            assert '\n'.join(filename2codelines[sampled_chunk['filename']][sampled_chunk['start_i']:sampled_chunk['end_i']]) == sampled_chunk['text']
        except:
            continue

        sampled_targets.append(sampled_chunk)

    # step 4: get cfc
    # use ThreadPool here as it seems sharing the large obj across processes is very slow. 
    pool = ThreadPool(args.num_processes)

    output_examples = []
    worker = partial(get_cfc, args=args, filename2codelines=filename2codelines, 
                     all_cfc_dict=all_cfc_dict, window_stride_configs=window_stride_configs)
    with tqdm(total=len(sampled_targets), desc=f'{repo_name}: collecting cfc') as pbar:
        for out in pool.imap_unordered(worker, sampled_targets):
            if len(out) > 0:
                output_examples.append(out)
            pbar.update()

    return output_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="json containing the repositories and files"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="language for the parser"
    )
    parser.add_argument(
        "--poisson_lambda",
        type=float,
        default=1.0,
        help="lambda value for poisson distribution"
    )
    parser.add_argument(
        "--cluster_ratio",
        type=float,
        default=0.2,
        help="number of clusters for each repo"
    )
    parser.add_argument(
        "--maximum_chunk_to_cluster",
        type=int,
        default=1000,
        help="max chunks to consider to cluster via KMeans"
    )
    parser.add_argument(
        "--topk_cfc",
        type=int,
        default=3,
        help="number of cross-file contexts to keep"
    )
    parser.add_argument(
        "--oracle_in_query",
        action='store_true',
        help="Use oracle target line in the query"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=20,
        help="number of processes to use"
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=200,
        help="number of repos to form a shard"
    )

    args = parser.parse_args()

    assert 'jsonl' in args.input_json

    parallel_procs = str(args.num_processes)
    os.environ["OMP_NUM_THREADS"] = parallel_procs
    os.environ["MKL_NUM_THREADS"] = parallel_procs
    os.environ["OPENBLAS_NUM_THREADS"] = parallel_procs
    os.environ["VECLIB_MAXIMUM_THREADS"] = parallel_procs
    os.environ["NUMEXPR_NUM_THREADS"] = parallel_procs

    mp.set_start_method('fork')

    with open(args.input_json) as f:
        data = [json.loads(line) for line in f.readlines()]

    accumulated_data, shard_id = [], 1
    for i_repo, repo in enumerate(data):
        output_examples = sample_chunks_with_cfc(args, repo, repo['repo_name'])
        accumulated_data.append(output_examples)
        if (i_repo > 0 and (i_repo+1) % args.shard_size == 0) or i_repo == len(data) - 1:
            accumulated_data = [y for x in accumulated_data for y in x]

            if args.oracle_in_query:
                outfilename = args.input_json.replace('.jsonl', f'.chunks.oracleinquery.shard{shard_id}.jsonl')
            else:
                outfilename = args.input_json.replace('.jsonl', f'.chunks.shard{shard_id}.jsonl')

            with open(outfilename, "w") as fw:
                for ex in accumulated_data:
                    fw.write(json.dumps(ex))
                    fw.write("\n")  
                    fw.flush()
                print('Dumped shard {} to {}'.format(shard_id, outfilename))

            accumulated_data, shard_id = [], shard_id + 1
