import re
import sys
import json
import timeout_decorator
import numpy as np

from tqdm import tqdm
from typing import Callable, List
from fuzzywuzzy import fuzz
import editdistance
from functools import partial
import torch.multiprocessing as mp
from tree_sitter import Language, Parser
from typing import List, Callable, Union
from tree_sitter.binding import Node as TSNode

parser = None


def cal_edit_sim(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        edit_sim += fuzz.ratio(pred, gt)
    return edit_sim / total


def cal_edit_sim_repoeval(references, hypotheses):
    total = len(references)
    edit_sim = 0.0
    for pred, gt in zip(hypotheses, references):
        pred = pred.strip()
        gt = gt.strip()
        if max(len(pred), len(gt)) == 0:
            continue
        edit_sim += 1 - editdistance.eval(pred, gt) / max(len(pred), len(gt))
    return edit_sim / total


def tokenize_code(code):
    code = re.sub(r"([^A-Za-z0-9_])", r" \1 ", code)
    code = re.sub(r"([a-z])([A-Z])", r"\1 \2", code)
    code = re.sub(r"\s+", " ", code)
    code = code.replace('"', "`")
    code = code.replace("'", "`")
    tokens = [t for t in code.split(" ") if t]
    return tokens


def cal_exact_match(references, hypotheses):
    em_score = []
    for pred, gold in zip(hypotheses, references):
        em_score.append(tokenize_code(pred) == tokenize_code(gold))
    return np.mean(em_score)


def remove_comments(code):
    code = re.sub(r'#.*', '', code)
    return code


def is_parse_valid(parser, code):
    def syntax_error(node):
        if node.type == "ERROR":
            return True
        try:
            for child in node.children:
                if syntax_error(child):
                    return True
        except RecursionError as err:
            return True

        return False

    tree = get_ast(parser, code)
    if tree is not None:
        return not syntax_error(tree.root_node)
    return False


def get_valid_completion(prompt, completion, parser):
    for i in range(len(completion), -1, -1):
        code = prompt + completion[:i]
        if is_parse_valid(parser, code):
            return "parseable", completion[:i].rstrip()

    return "not_parseable", completion


def dfs(
        node: TSNode,
        node_types: List[str],
        callback: Callable,
        ignore_node_types: List[str] = None,
):
    """
    Helper to traverse parsed AST
    """
    if node.type in node_types:
        callback(node)

    for child in node.children:
        if not ignore_node_types or child.type not in ignore_node_types:
            dfs(child, node_types, callback, ignore_node_types)


def collect_nodes(root_node, node_types, ignore_node_types=None):
    """
    Collect all nodes that belong to certain types
    """
    result = list()

    def _cb(n):
        result.append(n)

    if root_node is not None:
        try:
            dfs(root_node, node_types, _cb, ignore_node_types)
        except RecursionError as err:
            print('collection of nodes failed due to RecursionError')
            return []

    return result

@timeout_decorator.timeout(5)
def get_ast(parser, code):
    assert isinstance(code, str) or isinstance(code, bytes)
    if isinstance(code, str):
        code = bytes(code, "utf8")
    try:
        tree = parser.parse(code)
        return tree
    except Exception as e:
        return None


def get_functions(parser, code):
    """
    This function returns all functions (irrespective of whether they are inside a class) in a dict format.
    :param code:
    :return: Dict()
    """
    try:
        tree = get_ast(parser, code)
    except:
        return []
    if tree is None:
        return []

    functions = []
    function_nodes = collect_nodes(tree.root_node, ["function_definition"])
    for fnode in function_nodes:
        assert fnode.children[-1].type == "block"
        body_text = fnode.children[-1].text.decode("utf-8")
        functions.append(body_text)

    return functions


def get_function_completion(prompt, completion, parser):
    code = prompt + "pass"
    target_fn_idx = len(get_functions(parser, code)) - 1
    # assert target_fn_idx != -1

    code = prompt + completion
    function_body = get_functions(parser, code)[target_fn_idx]
    return function_body


def process_examples(task, args):
    sample, ex = args
    global parser

    prediction = sample["pred"]
    target = ex["groundtruth"]

    if task == "function_completion":
        status, prediction = get_valid_completion(ex["prompt"], prediction, parser)
        if status == "parseable":
            try:
                prediction = get_function_completion(ex["prompt"], prediction, parser)
                target = get_function_completion(ex["prompt"], target, parser)
            except:
                print(f'[warning] parsing failed: task_id:{ex["task_id"]}')
        else:
            print(f'[warning] parsing failed: task_id:{ex["task_id"]}')
    else:
        num_target_lines = sum([1 for l in target.split("\n") if l.strip()])
        pred_lines = [l for l in prediction.split("\n") if l.strip()][:num_target_lines]
        prediction = "\n".join(pred_lines)

    trunc_s = {
        "task_id": sample["task_id"],
        "pred": prediction,
        "target": target
    }
    return trunc_s


def compute_metric_stmt(args):
    with open(f"{args.output_dir}/prediction.jsonl", "r") as f_pred:
        samples = []
        for l in f_pred.readlines():
            samples.append(json.loads(l))

    examples = {}
    with open(args.prompt_file, "r") as f_in:
        for l in f_in.readlines():
            ex = json.loads(l)
            if hasattr(args, "focused_repo") and args.focused_repo and args.focused_repo not in re.sub('/', '_', ex['metadata']['repository']):
                continue
            examples[ex["metadata"]["task_id"]] = {
                "task_id": ex["metadata"]["task_id"],
                "prompt": ex["prompt"],
                "groundtruth": ex["groundtruth"]
            }

    # assert len(samples) == len(examples), f"{len(samples)} != {len(examples)}"
    if len(samples) == len(examples):
        print('Warning: len(samples) ({}) == len(examples) ({})'.format(len(samples), len(examples)))

    global parser
    # language = Language(args.ts_lib, "python")
    ts_lang = args.language
    if ts_lang == 'csharp':
        ts_lang = 'c_sharp'
    language = Language(args.ts_lib, ts_lang)
    parser = Parser()
    parser.set_language(language)

    truncated_samples = []
    print("post-processing samples ...")
    pool = mp.Pool(mp.cpu_count() - 1)
    worker = partial(process_examples, args.task)

    with tqdm(total=len(samples)) as pbar:
        for trunc_s in pool.imap_unordered(worker, zip(samples, [examples[s["task_id"]] for s in samples])):
            truncated_samples.append(trunc_s)
            pbar.update()

    with open(f"{args.output_dir}/prediction_truncated.jsonl", 'w', encoding="utf-8") as pt:
        for trunc_s in truncated_samples:
            pt.write(json.dumps(trunc_s) + "\n")

    ### Score calculation

    detailed_results = []
    exact_match = 0
    edit_sim = 0
    edit_sim_repoeval = 0

    for idx, trunc_s in enumerate(truncated_samples):
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        es_repoeval = cal_edit_sim_repoeval([trunc_s["target"]], [trunc_s["pred"]])
        em = cal_exact_match([trunc_s["target"]], [trunc_s["pred"]])
        edit_sim += es
        edit_sim_repoeval += es_repoeval
        exact_match += em

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em,
            "es": es,
            "es_repoeval": es_repoeval
        })

    em_ratio = round(exact_match / len(truncated_samples) * 100, 2)
    edit_sim = round(edit_sim / len(truncated_samples), 2)
    edit_sim_repoeval = round(edit_sim_repoeval / len(truncated_samples) * 100, 2)

    print(
        f"Code Matching: "
        f"EM {em_ratio:.2f}, "
        f"ES {edit_sim:.2f}, "
        f"ES RepoEval {edit_sim_repoeval:.2f}"
    )

    with open(f"{args.output_dir}/detailed_results.json", 'w') as f:
        for dr in detailed_results:
            f.write(json.dumps(dr) + "\n")

    # write the results to a file
    with open(f"{args.output_dir}/results.json", 'w') as f:
        res = {
            "em": em_ratio,
            "es": edit_sim,
            "es_repoeval": edit_sim_repoeval,
            "total": len(truncated_samples)
        }
        f.write(json.dumps(res, indent=2))


def compute_metric_stmt_custom(predictions_file, prompt_file, output_dir, 
                               ts_lib, task, focused_repo=None, anchor_file=None, out_f_suffix=""):
    eval_ids = set()

    if anchor_file:
        with open(anchor_file, "r") as f_pred:
            for l in f_pred.readlines():
                eval_ids.add(json.loads(l)['task_id'])

    with open(predictions_file, "r") as f_pred:
        samples = []
        for l in f_pred.readlines():
            if anchor_file:
                if json.loads(l)['task_id'] in eval_ids:
                    samples.append(json.loads(l))
            else:
                entry = json.loads(l)
                # entry['task_id'] = re.sub('-', '_',entry['task_id'])
                if entry['task_id'] in eval_ids:
                    continue
                if focused_repo is not None:
                    if type(focused_repo) == str and focused_repo not in re.sub('/', '_', entry['task_id']):
                        continue
                    elif type(focused_repo) == list and not any([x in re.sub('/', '_', entry['task_id']) for x in focused_repo]):
                        continue
                samples.append(entry)
                eval_ids.add(entry['task_id'])

    examples = {}
    with open(prompt_file, "r") as f_in:
        for l in f_in.readlines():
            ex = json.loads(l)
            if focused_repo is not None:
                if type(focused_repo) == str and focused_repo not in re.sub('/', '_', ex['metadata']['repository']):
                    continue
                elif type(focused_repo) == list and not any([x in re.sub('/', '_', ex['metadata']['repository']) for x in focused_repo]):
                    continue
            if ex["metadata"]["task_id"] not in eval_ids:
                continue
            examples[ex["metadata"]["task_id"]] = {
                "task_id": ex["metadata"]["task_id"],
                "prompt": ex["prompt"],
                "groundtruth": ex["groundtruth"]
            }

    assert len(samples) == len(examples), f"{len(samples)} != {len(examples)}"

    global parser
    language = Language(ts_lib, "python")
    parser = Parser()
    parser.set_language(language)

    truncated_samples = []
    print("post-processing samples ...")
    pool = mp.Pool(mp.cpu_count() - 1)
    worker = partial(process_examples, task)

    with tqdm(total=len(samples)) as pbar:
        for trunc_s in pool.imap_unordered(worker, zip(samples, [examples[s["task_id"]] for s in samples])):
            truncated_samples.append(trunc_s)
            pbar.update()

    with open(f"{output_dir}/prediction_truncated{out_f_suffix}.jsonl", 'w', encoding="utf-8") as pt:
        for trunc_s in truncated_samples:
            pt.write(json.dumps(trunc_s) + "\n")

    ### Score calculation

    detailed_results = []
    exact_match = 0
    edit_sim = 0
    edit_sim_repoeval = 0

    for idx, trunc_s in enumerate(truncated_samples):
        es = cal_edit_sim([trunc_s["target"]], [trunc_s["pred"]])
        es_repoeval = cal_edit_sim_repoeval([trunc_s["target"]], [trunc_s["pred"]])
        em = cal_exact_match([trunc_s["target"]], [trunc_s["pred"]])
        edit_sim += es
        edit_sim_repoeval += es_repoeval
        exact_match += em

        detailed_results.append({
            "task_id": trunc_s["task_id"],
            "em": em,
            "es": es,
            "es_repoeval": es_repoeval
        })

    em_ratio = round(exact_match / len(truncated_samples) * 100, 2)
    edit_sim = round(edit_sim / len(truncated_samples), 2)
    edit_sim_repoeval = round(edit_sim_repoeval / len(truncated_samples) * 100, 2)

    print(
        f"Code Matching: "
        f"EM {em_ratio:.2f}, "
        f"ES {edit_sim:.2f}, "
        f"ES RepoEval {edit_sim_repoeval:.2f}"
    )

    with open(f"{output_dir}/detailed_results{out_f_suffix}.json", 'w') as f:
        for dr in detailed_results:
            f.write(json.dumps(dr) + "\n")

    # write the results to a file
    with open(f"{output_dir}/results{out_f_suffix}.json", 'w') as f:
        res = {
            "em": em_ratio,
            "es": edit_sim,
            "es_repoeval": edit_sim_repoeval,
            "total": len(truncated_samples)
        }
        f.write(json.dumps(res, indent=2))


if __name__ == '__main__':
    ts_lib = '/home/diwu/code/Repoformer/repo_eval/build/python-lang-parser.so'
    ts_lib = '/home/diwu/code/Repoformer/repo_eval/build/python-lang-parser.so'
    # 'function_completion'   'api_completion'   'line_completion'
    # for task in ['line_completion', 'api_completion', 'function_completion']:
    # for task in ['function_completion']:
    # for task in ['api_completion']:
    for task in ['api_completion']:
        # for model in ['bigcode_starcoderbase_1b']:
        # for model in ['bigcode_starcoder']:
        # for model in ['salesforce_codegen_350m_mono', 'salesforce_codegen_2b_mono', 'salesforce_codegen_6b_mono', 'salesforce_codegen_16b_mono']:
        for model in [2, 3, 4, 5]:
        # for model in ['placeholder']:
            print(task, model)

            # focused_repo = ['alibaba_FederatedScope', 'awslabs_fortuna', 'google_vizier', 'huggingface_diffusers', 
            #                 'huggingface_evaluate', 'pytorch_rl', 'nerfstudio-project_nerfstudio', 'opendilab_ACE']  # line and API 
            # focused_repo = ['awslabs_fortuna', 'huggingface_diffusers', 
            #                 'pytorch_rl', 'nerfstudio-project_nerfstudio', 'opendilab_ACE']  # line and API 
            focused_repo = None
            if focused_repo is not None:
                print('Warning: focused_repo set to', focused_repo)
            
            # evaluating a single file
            anchor_file = None 

            # prompt_file = f'/mnt/efs/people/diwun/workspace/repoformer-v1/repo_eval/processed_data/python_{task}.jsonl'
            # prompt_file = f'/home/ec2-user/repoformer-v1/finetuning/data/helpfulness_simulation/test_0814_e0602_processed_api_completion.jsonl'
            # prompt_file = f'/mnt/efs/people/diwun/workspace/repoformer-v1/finetuning/data/stack-python/chunks_for_func_completion/shard59/with_oracle/9/sample_for_completion_merged.jsonl'
            # prompt_file = f'/local2/diwu/code/data/repoformer-330k-7b/labeling/function/sample_for_completion.jsonl'
            prompt_file = f'/home/diwu/code/Repoformer/repo_eval/processed_data/python_{task}.jsonl'
            
            # output_dir = f'/mnt/efs/people/diwun/workspace/repoformer-v1/results/repoeval/rcfcl_rg1/unixcoder/{task}/{model}/'
            # output_dir = f'/home/ec2-user/repoformer-v1/finetuning/data/temp_20230816/logs/lrcontext/{model}/'
            # output_dir = f'/mnt/efs/people/diwun/workspace/repoformer-v1/finetuning/data/stack-python/chunks_for_func_completion/shard59/with_oracle/9/logs/rcfcl_rg1/{model}/'
            # output_dir = f'/mnt/efs/people/diwun/workspace/repoformer-v1/results/repoeval/rcfcl_rg1/sparse/{task}/{model}/'
            # output_dir = '/local2/diwu/code/data/repoformer-330k-7b/labeling/function/logs/rcfcl_rg1/sparse/starcoderbase-7b/'
            output_dir = f'/home/diwu/code/Repoformer/results/repoeval/rcfcl_rg1/sparse/{task}/deepmind_specdec/TGTbigcode_starcoderbase_3b_DFTbigcode_starcoderbase_1b_LOOKAHEAD{model}_TEMP0.2_TOPP0.95/'
            # output_dir = f'/home/diwu/code/Repoformer/results/repoeval/rcfcl_rg1/sparse/{task}/deepmind_specdec/TGTbigcode_starcoderbase_3b_DFTrepoformer_1b_lrcfc_LOOKAHEAD{model}_TEMP0.2_TOPP0.95/'
            # output_dir = f'/home/diwu/code/Repoformer/results/repoeval/rcfcl_rg1/sparse/{task}/deepmind_specdec/TGTbigcode_starcoderbase_7b_autoregressive_TEMP0.2_TOPP0.95/'
            
            predictions_file = f'{output_dir}/prediction.jsonl'

            compute_metric_stmt_custom(predictions_file, prompt_file, output_dir, ts_lib, task, focused_repo=focused_repo, anchor_file=anchor_file, out_f_suffix="")
            
            # some code for evaluating knn
            '''
            knn_lambda = 0.25
            #knn_temp = 0.7
            for focused_repo in ['alibaba_FederatedScope', 'awslabs_fortuna', 'google_vizier', 'huggingface_diffusers', 'huggingface_evaluate', 'pytorch_rl', 'nerfstudio-project_nerfstudio', 'opendilab_ACE']:
                # for focused_repo in ['huggingface_diffusers']:
                # try:
                #     prompt_file = f'/home/ec2-user/repoformer/repo_eval/processed_data/python_{task}.jsonl'
                #     anchor_file = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/prediction.jsonl'
                #     predictions_file = f'/home/ec2-user/repoformer/results/repoeval/knnlm/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/prediction.jsonl'
                #     output_dir = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/'
                #     compute_metric_stmt_custom(predictions_file, prompt_file, output_dir, ts_lib, task, focused_repo=None, anchor_file=anchor_file, out_f_suffix="_knnoragref")
                #     print(output_dir, 'knn no rag baseline')
                # except:
                #     continue

                # try:
                #     prompt_file = f'/home/ec2-user/repoformer/repo_eval/processed_data/python_{task}_sparse_rg1.jsonl'
                #     anchor_file = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/prediction.jsonl'
                #     predictions_file = f'/home/ec2-user/repoformer/results/repoeval/rg1/sparse/{task}/{model}/prediction.jsonl'
                #     output_dir = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/'
                #     compute_metric_stmt_custom(predictions_file, prompt_file, output_dir, ts_lib, task, focused_repo=None, anchor_file=anchor_file, out_f_suffix="_sparserg1ref")
                #     print(output_dir, 'sparse rg1 baseline')
                # except:
                #     continue

                try:
                    prompt_file = f'/home/ec2-user/repoformer/repo_eval/processed_data/python_{task}.jsonl'
                    anchor_file = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/prediction.jsonl'
                    predictions_file = f'/home/ec2-user/repoformer/results/repoeval/baseline/{task}/{model}/prediction.jsonl'
                    output_dir = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/'
                    compute_metric_stmt_custom(predictions_file, prompt_file, output_dir, ts_lib, task, focused_repo=None, anchor_file=anchor_file, out_f_suffix="_zsref")
                    print(output_dir, 'zs baseline')
                except:
                    continue

                try:
                    prompt_file = f'/home/ec2-user/repoformer/repo_eval/processed_data/python_{task}_sparse_rg1.jsonl'
                    anchor_file = None 
                    predictions_file = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/prediction.jsonl'
                    # predictions_file = f'/home/ec2-user/repoformer/results/repoeval/knnlm_leftrightcontext_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024_temperature{knn_temp}/{focused_repo}/{task}/{model}/prediction.jsonl' 
                    # output_dir = f'/home/ec2-user/repoformer/results/repoeval/knnlm_leftrightcontext_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024_temperature{knn_temp}/{focused_repo}/{task}/{model}/'
                    output_dir = f'/home/ec2-user/repoformer/results/repoeval/knnlm_sparse_rg1_{task}/codegen-350m/knn_lambda{knn_lambda}_k1024/{focused_repo}/{task}/{model}/'
                    compute_metric_stmt_custom(predictions_file, prompt_file, output_dir, ts_lib, task, focused_repo=None, anchor_file=anchor_file, out_f_suffix="")
                    print(output_dir, 'evaluated exp')
                except:
                    continue
            '''
