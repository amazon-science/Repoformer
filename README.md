# Repoformer
This repository contains the data and inference code of the ICML 2024 paper "[Repoformer: Selective Retrieval for
Repository-Level Code Completion.](https://arxiv.org/abs/2403.10059)"

Work done by Di Wu, Wasi Uddin Ahmad, Dejiao Zhang, Murali Krishna Ramanathan, Xiaofei Ma.
 

## Requirements

- Install all dependencies: `pip install -r requirements.txt`
- Build tree sitter: `bash scripts/build_treesitter.sh`
- Prepare RepoEval data: 
    - `cd repo_eval/data`
    - `bash download.sh`
    - `bash prepare.sh`
    - `cd ../cfc_retrieval`
    - `bash run.sh`  (Use Jaccard similarity by default. Uncomment the other lines to use other retrievers.)
- Prepare CrossCodeEval data: 
    - `cd cceval/data`
    - `bash prepare_data.sh`

## Training data creation
We start from preprocessed repositories from the stack. To reproduce our data creation strategy, you can prepare the repositories in the following format:
```
{
    "repo_name": "...",
    "stars_count": 100,
    "files": [
        {
            "filepath": "...",
            "content": "",
            "metadata": {
                "size": 100,
                "lang": "Python",
                "ext": "py",
                "hexsha": "...",
                "avg_line_length": 22,
                "max_line_length": 47,
                "line_count": 19,
                "non_empty_line_count": 16,
                "imports": [
                    "..."   # all import names
                ],
                "local_imports": [
                    "..."   # local import names
                ]
            }
        },
        ...  # one entry per file
    ]
     "repo_size": {
        "number_of_files": 30,
        "lines_of_code": 900
    }
}
```

Starting from the raw file `raw.jsonl`, the data sampling algorithm contains three steps: blank sampling, RAG simulation, and data merging. 

#### Step 1: blank sampling
```
cd finetuning/data_creation/

# for creating chunk completion data
python 1_create_chunk.py --lang [python/java/csharp/javascript] --input_json raw.jsonl --poisson_lambda 3.0 --num_processes 20 --cluster_ratio 0.1 --shard_size 500 [--oracle_in_query]

# for creating function completion data
python 1_create_function.py --lang [python/java/csharp/javascript] --input_json raw.jsonl --poisson_lambda 3.0 --num_processes 20 --cluster_ratio 0.1 --shard_size 500 [--oracle_in_query]
```
Note that the `--oracle_in_query` flag uses the target line for retrieving the relevant contexts. In the paper, half of the data is created with `--oracle_in_query` and half is created without it. We output data in shards to make downstream processing easier. The `--shard_size` parameter controls the size of each shard.

#### Step 2: RAG simulation
Suppose the previous step's outputs are named as `chunk_shardx/sample_for_completion.jsonl` and `function_shardx/sample_for_completion.jsonl`. To obtain the label for Repoformer, we run inference *twice*: once with the retrieved context and once without. 

```
cd finetuning/data_creation/2_labeling

# for labeling the chunk completion data
bash run_chunk_lrcontext.sh starcoderbase-1b chunk_shardx/ 
bash run_chunk_rcfcl_rg1.sh starcoderbase-1b chunk_shardx/

# for labeling the function completion data
bash run_function_lrcontext.sh starcoderbase-1b function_shardx/ 
bash run_function_rcfcl_rg1.sh starcoderbase-1b function_shardx/

```

#### Step 3: data merging
After step 2, the model outputs and scores should be stored in `chunk_shardx/logs` and `function_shardx/logs` . You can run the following command to get the final data.

```
# For generating the final chunk completion data. Function completion data is similar.

python 3_generate_labelled_data.py --raw_file chunk_shardx/sample_for_completion.jsonl --baseline_scores_file chunk_shardx/logs/lrcontext/starcoderbase-1b/detailed_results.json --rg1_scores_file chunk_shardx/logs/rcfcl_rg1/sparse/starcoderbase-1b/detailed_results.json --output_file chunk_shardx/data_labelled.jsonl --generation_model starcoderbase-1b 
```


## Training Repoformer
Our training code is based on [ContraCLM](https://github.com/amazon-science/ContraCLM). 

#### Step 1: tokenization
We tokenize the data into arrow format datasets. To run the code, move the files from step 3 into a separate folder and provide its path in `finetuning/preprocess/run_preprocess_repoformer_cfcinrc.sh`. Then, run the following command:
```
cd finetuning/preprocess
bash run_preprocess_repoformer_cfcinrc.sh
```
Note that in this repo, `<end_rc>` corresponds to the `<eof>` token in the paper, and `<cfc_info>` corresponds to `<cc>`. Repoformer only need to add these two special tokens. 

#### Step 2: running training
Before running the script, make sure to update `finetuning/runscripts/run_repoformer_final_setting.sh` with your preprocessed data path.
```
cd finetuning/
bash runscripts/run_repoformer_final_setting.sh
```

#### Step 3: checkpoint postprocessing
After training, the deepspeed checkpoint will be stored in the `last.ckpt` folder. You can get the checkpoint in huggingface format with the following steps:
- `cd /path/to/last.ckpt/`
- `python zero_to_fp32.py . pytorch_model.bin.original`
- Update `finetuning/evaluation/process_checkpoint_state_dict.py` to point to the StarCoder model with the correct size.
- `python finetuning/evaluation/process_checkpoint_state_dict.py /path/to/last.ckpt/`

## Evaluation
#### Datasets
We release the newly created CrossCodeLongEval benchmark under the folder `crosscodelongeval`. You may run the `process_data.sh` to preprocess the data. In addition, we release the code to download and use Repoeval/CrossCodeEval in the folders `repo_eval` and `cceval`.

#### Baselines
To get the results of the baselines with or without repository-level retrieval, we recommend using the `run_fim_hf.sh` in the `repo_eval` and `cceval` folder. Sample command:
```
bash run_fim_hf.sh model exp retriever
```
- `model`: We support `starcoderbase-1b/3b/7b` and `starcoder`. You can easily evaluate on other code LMs you like by changing the model name. Note that if the LM does not perform fill-in-the-middle generation, the `--use_fim_prompt` flag needs to be dropped.
- `exp`: the prompting strategy. There are four possible settings. `lrcontext` and `rcfcl_rg1` are the two settings used in the Repoformer paper.
    - `baseline`: left context only.
    - `lrcontext`: left context + right context.
    - `rg1`: left context + retrieved cross-file context.
    - `rcfcl_rg1`: left context + right context + retrieved cross-file context.    
- `retriever`: 
    - For RepoEval, we support `sparse` (Jaccard similarity) and `unixcoder`.
    - For CCEval and CrossCodeLongEval, we support `bm25`, `openai_cosine_sim`, and `unixcoder_cosine_sim`.

We also support vllm for inference. For vllm, you would need torch 2.x. The other requirements are the same as in `requirements.txt`. 

#### Repoformer
After converting the checkpoint, you can run the evaluation directly using the followng commands. 
```
cd finetuning/evaluation

# evaluate on RepoEval
bash run_repoeval.sh

# evaluate on CCEval
bash run_cceval.sh
```
