import os
import json
import time
import glob
import argparse
import multiprocessing as mp

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from utils import file_distance, tokenize_nltk
from rerank_utils import lexical_ranking, SemanticReranking

# repocoder_packages.jsonl data format

# "source": "github",
# "url": "https://github.com/alibaba/FederatedScope",
# "license": "permissive",
# "patent": "NO_PATENT_INFO",
# "repository": "https://github.com/alibaba/FederatedScope",
# "filepath": "alibaba_FederatedScope/setup.py",
# "content": [
#     {
#         "type": "code",
#         "lang": "python",
#         "content": "..."
#     }
# ]


def get_importance_weights_v1(query, query_type):
    query_list = query.split('\n')
    weights = []
    if query_type == 'left_to_hole':
        for i in range(len(query_list)):
            weights.append(1.0 / (i + 1))
        weights = weights[::-1]
    elif query_type == 'right_to_hole':
        for i in range(len(query_list)):
            weights.append(1.0 / (i + 1))
    else:
        raise NotImplementedError

    assert len(query_list) == len(weights)
    return weights


def get_importance_weights_v2(query, query_type):
    query_list = query.split('\n')
    weights = []
    if query_type == 'left_to_hole':
        for i in range(len(query_list)):
            weights.append(1.0 / np.exp(i))
        weights = weights[::-1]
    elif query_type == 'right_to_hole':
        for i in range(len(query_list)):
            weights.append(1.0 / np.exp(i))
    else:
        raise NotImplementedError

    assert len(query_list) == len(weights)
    return weights


def get_crossfile_context_from_chunks(
        args,
        prompt,
        code_chunks,
        code_chunk_ids,
        groundtruth,
        semantic_ranker,
        repocoder_pred=None,
        right_context=None
):
    """
    Returns a set of file chunks as crossfile context.

    :param args:
    :param prompt:
    :param code_chunks:
    :param code_chunk_ids:
    :param groundtruth:
    :return:
    """
    
    start = time.time()

    assert len(code_chunks) != 0
    tr_code_chunks = code_chunks[:args.maximum_chunk_to_rerank]
    tr_code_chunk_ids = code_chunk_ids[:args.maximum_chunk_to_rerank]

    importance_weights = None
    if args.query_type == "groundtruth":
        # oracle experiment
        prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
        groundtruth_lines = [gt for gt in groundtruth.split("\n") if gt.strip()]
        code_lines = prompt_lines + groundtruth_lines
        query = "\n".join(code_lines[-args.query_length:])
    elif args.query_type == "last_n_lines":
        prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
        query = "\n".join(prompt_lines[-args.query_length:])
        # if 'line_by_line' in args.ranking_fn:
        #     importance_weights = get_importance_weights(query, query_type='left_to_hole')
    elif args.query_type == "first_n_lines":
        assert right_context is not None
        right_context_lines = [pl for pl in right_context.split("\n") if pl.strip()]
        query = "\n".join(right_context_lines[:args.query_length])
        # if 'line_by_line' in args.ranking_fn:
        #     importance_weights = get_importance_weights(query, query_type='left_to_hole')
    elif args.query_type == "repocoder":
        prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
        pred_lines = [pl for pl in repocoder_pred.split("\n") if pl.strip()]
        if args.use_lr_context_repocoder:
            right_context_lines = [pl for pl in right_context.split("\n") if pl.strip()]
            left_n_lines = args.query_length // 2
            pred_n_lines = min(len(pred_lines), args.repocoder_hyp_n_lines_to_use)
            right_n_lines = args.query_length - left_n_lines - pred_n_lines
            query = "\n".join(prompt_lines[-left_n_lines:]
                              + pred_lines[:pred_n_lines]
                              + right_context_lines[:right_n_lines])
        else:
            prompt_keep_line = args.query_length - min(args.repocoder_query_hyp_size, len(pred_lines))
            query = "\n".join(prompt_lines[-prompt_keep_line:] + pred_lines[:args.repocoder_query_hyp_size])
    elif args.query_type == "left_last_and_right_first":
        assert right_context is not None
        left_n_lines = args.query_length // 2
        right_n_lines = args.query_length - left_n_lines
        prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
        right_context_lines = [pl for pl in right_context.split("\n") if pl.strip()]
        query = "\n".join(prompt_lines[-left_n_lines:] + right_context_lines[:right_n_lines])
        importance_weights = []
        if 'line_by_line' in args.ranking_fn:
            if args.ranking_fn == 'bm25_line_by_line_weighted2':
                importance_weights += get_importance_weights_v2('\n'.join(prompt_lines[-left_n_lines:]), query_type='left_to_hole')
                importance_weights += get_importance_weights_v2('\n'.join(right_context_lines[:right_n_lines]), query_type='right_to_hole')
            elif args.ranking_fn == 'bm25_line_by_line_weighted1':
                importance_weights += get_importance_weights_v1('\n'.join(prompt_lines[-left_n_lines:]), query_type='left_to_hole')
                importance_weights += get_importance_weights_v1('\n'.join(right_context_lines[:right_n_lines]), query_type='right_to_hole')
    else:
        raise NotImplementedError

    if args.ranking_fn == "cosine_similarity" or args.ranking_fn == "cosine_similarity_tokenwise":
        gpu_id = int(mp.current_process().name.split('-')[-1]) - 1
        # gpu_id = int(mp.current_process().name.split('-')[-1]) // torch.cuda.device_count() # - 1
        tr_code_chunks, tr_code_chunk_ids, ranking_scores = semantic_ranker.rerank(
            query, tr_code_chunks, tr_code_chunk_ids, gpu_id
        )
    else:
        tr_code_chunks, tr_code_chunk_ids, ranking_scores = lexical_ranking(
            query, tr_code_chunks, args.ranking_fn, tr_code_chunk_ids, 
            importance_weights=importance_weights
        )

    ranking_latency = time.time() - start
    num_ranking_candidate = len(tr_code_chunks)

    top_k = min(args.use_topk_chunks, len(tr_code_chunk_ids))
    if top_k == 0:
        return "", ranking_latency, num_ranking_candidate

    selected_chunks = []
    selected_chunks_filename = []
    selected_chunks_scores = []

    if args.use_next_chunk_as_cfc:
        # prepare an id2idx map
        assert len(tr_code_chunks) == len(tr_code_chunk_ids)
        id2idx = dict()
        for j, cci in enumerate(code_chunk_ids):
            id2idx[cci] = j
        next_chunk_not_found = 0
        for cidx, _id in enumerate(tr_code_chunk_ids[:top_k]):
            fname, c_id = _id.rsplit("|", 1)
            next_id = f"{fname}|{int(c_id) + 1}"
            if next_id not in id2idx:

                next_chunk_not_found += 1
                to_add = code_chunks[id2idx[_id]]
            else:
                to_add = code_chunks[id2idx[next_id]]

            if to_add not in selected_chunks:
                selected_chunks.append(to_add)
                selected_chunks_filename.append(fname)
                selected_chunks_scores.append(ranking_scores[cidx])

        # if next_chunk_not_found > 0:
        #     print(f"[Warning] For {next_chunk_not_found} out of {len(selected_chunks)} "
        #           f"chunks, next chunks was not found.")
    elif args.use_last_chunk_as_cfc:
        # prepare an id2idx map
        assert len(tr_code_chunks) == len(tr_code_chunk_ids)
        id2idx = dict()
        for j, cci in enumerate(code_chunk_ids):
            id2idx[cci] = j
        last_chunk_not_found = 0
        for cidx, _id in enumerate(tr_code_chunk_ids[:top_k]):
            fname, c_id = _id.rsplit("|", 1)
            last_id = f"{fname}|{int(c_id) - 1}"
            if last_id not in id2idx:

                last_chunk_not_found += 1
                to_add = code_chunks[id2idx[_id]]
            else:
                to_add = code_chunks[id2idx[last_id]]

            if to_add not in selected_chunks:
                selected_chunks.append(to_add)
                selected_chunks_filename.append(fname)
                selected_chunks_scores.append(ranking_scores[cidx])
        # if next_chunk_not_found > 0:
        #     print(f"[Warning] For {next_chunk_not_found} out of {len(selected_chunks)} "
        #           f"chunks, next chunks was not found.")
    else:
        selected_chunks = tr_code_chunks[:top_k]
        selected_chunks_filename = [_id.rsplit("|", 1)[0] for _id in tr_code_chunk_ids[:top_k]]
        selected_chunks_scores = ranking_scores[:top_k]

    line_start_sym = "#"
    cfc = f"{line_start_sym} Here are some relevant code fragments from other files of the repo:\n\n"
    for sc, scf in zip(selected_chunks, selected_chunks_filename):
        cfc += f"{line_start_sym} the below code fragment can be found in:\n{line_start_sym} {scf}" + "\n"
        cfc += "\n".join([f"{line_start_sym} {cl}" for cl in sc.strip('\n').splitlines()]) + "\n\n"

    return cfc, ranking_latency, num_ranking_candidate


def find_files_within_distance_k(current_file_path, filelist, k):
    list_of_modules = []
    module_weight = []
    for filepath in filelist:
        if filepath != current_file_path:
            dist = file_distance(filepath, current_file_path)
            if dist == -1:
                continue
            elif dist <= k:
                list_of_modules.append(filepath)
                module_weight.append(dist)

    # sorting in ascending order
    list_of_modules = [x for _, x in sorted(zip(module_weight, list_of_modules))]
    return list_of_modules


def get_cfc(example, args, semantic_ranker, repositories):
    project_context = repositories[example["metadata"]["repository"]]

    status = None
    if len(project_context) == 0:
        example["crossfile_context"] = ""
        status = f"project_not_found"
    else:
        current_filecontent = None
        for filepath, content in project_context.items():
            if filepath == example["metadata"]["filepath"]:
                current_filecontent = content
                break

        if current_filecontent is None:
            example["crossfile_context"] = ""
            status = "file_not_found_in_project"

        else:
            pyfiles = find_files_within_distance_k(
                example["metadata"]["filepath"],
                list(project_context.keys()),
                k=args.crossfile_distance
            )
            pyfiles = pyfiles[:args.maximum_cross_files]

            code_chunks = []
            code_chunk_ids = []
            for pyfile in pyfiles:
                lines = project_context[pyfile].split("\n")
                lines = [l for l in lines if l.strip()]  # removing empty lines
                c_id = 0
                for i in range(0, len(lines), args.sliding_window_size):
                    c = "\n".join(lines[i:i + args.chunk_size])
                    tokenized_c = tokenize_nltk(c)
                    if len(tokenized_c) > 0:
                        code_chunks.append(c)
                        code_chunk_ids.append(f"{pyfile}|{c_id}")
                        c_id += 1

            if len(code_chunks) == 0:
                example["crossfile_context"] = ""
                status = "no_crossfile_context"

            else:
                cfc, rl, num_cands = get_crossfile_context_from_chunks(
                    args=args,
                    prompt=example["prompt"],
                    code_chunks=code_chunks,
                    code_chunk_ids=code_chunk_ids,
                    groundtruth=example["groundtruth"],
                    semantic_ranker=semantic_ranker,
                    repocoder_pred=None if not args.is_repocoder else example["pred"],
                    right_context=None if not args.query_type in ['first_n_lines', 'left_last_and_right_first', 'repocoder'] else example["right_context"],
                )
                example["crossfile_context"] = cfc
                if args.log_latency:
                    example["ranking_latency"] = rl

    return example, status


def attach_data(args):
    repositories = dict()
    with open(args.repository_file) as f:
        for line in f:
            ex = json.loads(line.strip())
            if ex["repository"] not in repositories:
                repositories[ex["repository"]] = dict()
            repositories[ex["repository"]][ex["filepath"]] = ex["content"][0]["content"]

    empty_cfc = 0
    error_freq = {
        "project_not_found": 0,
        "file_not_found_in_project": 0,
        "no_crossfile_context": 0
    }
    output_examples = []

    base_model_outputs = {}
    if args.is_repocoder:
        with open(args.base_model_pred_file) as f:
            for line in f.readlines():
                ex = json.loads(line.strip())
                base_model_outputs[ex["task_id"]] = ex["pred"]

    examples = []
    with open(args.input_file) as f:
        for line in f:
            ex = json.loads(line)
            if args.is_repocoder:
                ex["pred"] = base_model_outputs[ex["metadata"]["task_id"]]
            examples.append(ex)
            # ex = {
            #     "prompt": prompt,
            #     "groundtruth": groundtruth,
            #     "right_context": right_context,
            #     "metadata": {
            #         "task_id": "huggingface_diffusers/0",
            #         "filepath": "huggingface_diffusers/tests/pipelines/stable_diffusion/test_stable_diffusion_inpaint_legacy.py",
            #         "repository": "https://github.com/huggingface/diffusers",
            #         "url": "https://github.com/huggingface/diffusers"
            #     }
            # }

    semantic_ranker = None
    if args.ranking_fn == "cosine_similarity":
        if args.dense_retriever_type == "codewhisperer" or args.dense_retriever_type == "codewhisperer_v2":
            semantic_ranker = SemanticReranking(
                args.dense_retriever_type,
                tokenizer_name=args.tokenizer_name,
                config_name=args.config_name,
                model_name_or_path=args.model_name_or_path
            )
        else:
            semantic_ranker = SemanticReranking(args.dense_retriever_type)
    elif args.ranking_fn == "cosine_similarity_tokenwise":
        if args.dense_retriever_type == "codewhisperer" or args.dense_retriever_type == "codewhisperer_v2":
            semantic_ranker = SemanticReranking(
                args.dense_retriever_type,
                tokenizer_name=args.tokenizer_name,
                config_name=args.config_name,
                model_name_or_path=args.model_name_or_path,
                sim_function='token_sim'
            )
        else:
            semantic_ranker = SemanticReranking(args.dense_retriever_type, sim_function='token_sim')

    pool = mp.Pool(args.num_processes)
    worker = partial(get_cfc, args=args, semantic_ranker=semantic_ranker, repositories=repositories)

    with tqdm(total=len(examples)) as pbar:
        for (d, stat) in pool.imap_unordered(worker, examples):
            if stat in error_freq:
                error_freq[stat] += 1
            if len(d["crossfile_context"]) == 0:
                empty_cfc += 1
            output_examples.append(d)
            pbar.update()

    print("Total examples with empty CFC: ", empty_cfc)
    print(error_freq)
    return output_examples


if __name__ == "__main__":
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="input jsonl file"
    )
    parser.add_argument(
        "--repository_file",
        type=str,
        required=True,
        help="input repository file"
    )
    parser.add_argument(
        "--ranking_fn",
        type=str,
        default="bm25",
        choices=["bm25", "jaccard_similarity", "cosine_similarity", "cosine_similarity_tokenwise", "weighted_ngram_match_score",
                "bm25_line_by_line", "bm25_line_by_line_weighted1", "bm25_line_by_line_weighted2"],
        help="ranking function"
    )
    parser.add_argument(
        "--dense_retriever_type",
        type=str,
        default="unixcoder",
        choices=["unixcoder", "codewhisperer", "codewhisperer_v2"],
        help="dense retriever model type"
    )
    parser.add_argument(
        "--query_type",
        type=str,
        default="last_n_lines",
        choices=["last_n_lines", "groundtruth", "repocoder", "left_last_and_right_first", "first_n_lines"],
        help="how to form query from prompt"
    )
    parser.add_argument(
        "--crossfile_distance",
        type=int,
        default=100,
        help="max distance to search for crossfile"
    )
    parser.add_argument(
        "--maximum_chunk_to_rerank",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--maximum_cross_files",
        type=int,
        default=1000,
        help="max chunks to consider to rank via BM25"
    )
    parser.add_argument(
        "--use_next_chunk_as_cfc",
        type=str2bool,
        default=True,
        help="use next code chunk as context"
    )
    parser.add_argument(
        "--use_last_chunk_as_cfc",
        type=str2bool,
        default=False,
        help="use last code chunk as context (only for right context as query)"
    )
    parser.add_argument(
        "--output_file_suffix",
        type=str,
        default=None,
        help="add a suffix string to the output file"
    )

    # for codewhisperer model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="path to pretrained model or model identifier from huggingface"
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="pretrained config name or path"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="pretrained tokenizer name or path"
    )

    # repocoder
    parser.add_argument(
        "--is_repocoder",
        action='store_true',
        help="repocoder setting (using model output to retrieve)"
    )
    parser.add_argument(
        "--repocoder_hyp_n_lines_to_use",
        type=int,
        default=1,
        help="lines of hypotheses to use in query (repocoder setting)"
    )
    parser.add_argument(
        "--base_model_pred_file",
        type=str,
        default=None,
        help="predictions from the base model (for RepoCoder)"
    )
    parser.add_argument(
        "--use_lr_context_repocoder",
        action='store_true',
        help="use both left and right context to retrieve (for repocoder)"
    )

    # for specifying chunk configurations
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10,
        help="size of each code chunk"
    )
    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=10,
        help="sliding stride size"
    )
    parser.add_argument(
        "--query_length",
        type=int,
        default=10,
        help="length of query (in number of lines from the end of the left context)"
    )
    parser.add_argument(
        "--use_topk_chunks",
        type=int,
        default=10,
        help="use top k chunks from the retrieved results"
    )
    parser.add_argument(
        "--repocoder_query_hyp_size",
        type=int,
        default=5,
        help="number of lines from hypotheses to include in the query (repocoder only)"
    )
    parser.add_argument(
        "--log_latency",
        action='store_true',
        help="measure the retrieval latency and store it to the result"
    )
    
    args = parser.parse_args()

    args.output_file_suffix = "" if args.output_file_suffix is None else f"_{args.output_file_suffix}"

    args.use_next_chunk_as_cfc = True
    if args.query_type in ["groundtruth", "left_last_and_right_first", "first_n_lines"]:
        args.use_next_chunk_as_cfc = False
    if args.is_repocoder:
        assert args.query_type == "repocoder"
        assert args.base_model_pred_file is not None
        args.use_next_chunk_as_cfc = False

    args.use_last_chunk_as_cfc = False
    if args.query_type in ['first_n_lines']:
        args.use_last_chunk_as_cfc = True
    
    # global CHUNK_SIZE, SLIDING_WINDOW_SIZE, REPOCODER_QUERY_HYP_SIZE, USE_TOPK_CHUNKS, QUERY_LENGTH

    # CHUNK_SIZE = args.chunk_size  # 10
    # SLIDING_WINDOW_SIZE = args.sliding_window_size  # 10  # non-overlapping chunks if SLIDING_WINDOW_SIZE=CHUNK_SIZE
    # REPOCODER_QUERY_HYP_SIZE = args.repocode_query_hyp_size  # 5   # lines in hypotheses to include in query
    # USE_TOPK_CHUNKS = args.use_topk_chunks  # 10
    # QUERY_LENGTH = args.query_length  # 10  # last N lines from prompt will be query
    assert args.repocoder_query_hyp_size <= args.query_length

    import os
    args.num_processes = 30
    parallel_procs = str(args.num_processes)
    os.environ["OMP_NUM_THREADS"] = parallel_procs
    os.environ["MKL_NUM_THREADS"] = parallel_procs
    os.environ["OPENBLAS_NUM_THREADS"] = parallel_procs
    os.environ["VECLIB_MAXIMUM_THREADS"] = parallel_procs
    os.environ["NUMEXPR_NUM_THREADS"] = parallel_procs

    if args.ranking_fn == "cosine_similarity" or args.ranking_fn == "cosine_similarity_tokenwise":
        num_gpus = torch.cuda.device_count()
        args.num_processes = num_gpus
    mp.set_start_method('spawn')

    output_examples = attach_data(args)
    if args.is_repocoder:
        output_dir = os.path.dirname(args.base_model_pred_file)
    else:
        output_dir = os.path.dirname(args.input_file)
    output_filename = file_basename = os.path.splitext(os.path.basename(args.input_file))[0]

    outfilename = os.path.join(output_dir, f"{output_filename}" + args.output_file_suffix + ".jsonl")
    with open(outfilename, "w") as fw:
        for ex in output_examples:
            fw.write(json.dumps(ex))
            fw.write("\n")
