import sys
import torch
from rank_bm25 import BM25Okapi
from typing import List
from multiprocessing import Pool, cpu_count
from utils import tokenize_nltk
import numpy as np
from transformers import AutoModel, AutoConfig, AutoTokenizer
from codebleu import calc_codebleu


def jaccard_similarity(tokenized_query, tokenized_doc, containment=False):
    set1 = set(tokenized_query)
    set2 = set(tokenized_doc)
    intersection = len(set1.intersection(set2))
    union = len(set1) if containment else len(set1.union(set2))
    return float(intersection) / union


def tokenize_corpus(corpus, tokenizer_fn):
    pool = Pool(cpu_count())
    tokenized_corpus = pool.map(tokenizer_fn, corpus)
    return tokenized_corpus


def tokenize_query_and_docs(query, docs):
    tokenized_query = tokenize_nltk(query)
    tokenized_docs = [tokenize_nltk(d) for d in docs]
    return tokenized_query, tokenized_docs


def lexical_ranking(
        query,
        docs,
        ranking_fn,
        doc_ids=None,
        score_threshold=None,
        importance_weights=None
):
    if ranking_fn == "bm25":
        tokenized_query, tokenized_docs = tokenize_query_and_docs(query, docs)
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)
    elif ranking_fn == "jaccard_similarity":
        tokenized_query, tokenized_docs = tokenize_query_and_docs(query, docs)
        scores = [jaccard_similarity(tokenized_query, d, containment=False) for d in tokenized_docs]
    elif ranking_fn == "bm25_line_by_line":
        queries = query.split("\n")
        tokenized_queries = [tokenize_nltk(q) for q in queries]
        tokenized_docs = [tokenize_nltk(d) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        scores = [bm25.get_scores(q) for q in tokenized_queries]
        scores = [sum(s) for s in zip(*scores)]
    elif ranking_fn in ["bm25_line_by_line_weighted1", "bm25_line_by_line_weighted2"]:
        queries = query.split("\n")
        tokenized_queries = [tokenize_nltk(q) for q in queries]
        tokenized_docs = [tokenize_nltk(d) for d in docs]
        bm25 = BM25Okapi(tokenized_docs)
        scores = [bm25.get_scores(q) for q in tokenized_queries]
        final_scores = [0 for _ in scores[0]]
        for i_s, scores_list in enumerate(scores):
            for j_s, s in enumerate(scores_list):
                final_scores[j_s] += s * importance_weights[i_s]
        scores = final_scores
    elif ranking_fn == "weighted_ngram_match_score":
        scores = [calc_codebleu([query], [d], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)['weighted_ngram_match_score'] for d in docs]
    else:
        raise NotImplementedError

    if score_threshold:
        skip_ids = [idx for idx, s in enumerate(scores) if s < score_threshold]
        scores = [s for idx, s in enumerate(scores) if idx not in skip_ids]
        docs = [d for idx, d in enumerate(docs) if idx not in skip_ids]
        if doc_ids is not None:
            doc_ids = [doc_id for idx, doc_id in enumerate(doc_ids) if idx not in skip_ids]

    docs = [x for _, x in sorted(zip(scores, docs), reverse=True)]
    if doc_ids is not None:
        doc_ids = [x for _, x in sorted(zip(scores, doc_ids), reverse=True)]
    return docs, doc_ids, scores


class SemanticReranking:

    def __init__(self, model_type="unixcoder", **kwargs):
        self.sim_function = kwargs.get("sim_function", "pooled_sim")
        self.model_type = model_type
        if self.model_type == "unixcoder":
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/unixcoder-base')
            self.model = AutoModel.from_pretrained('microsoft/unixcoder-base')
        elif self.model_type == "codewhisperer":
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs['tokenizer_name'])
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            config = AutoConfig.from_pretrained(kwargs['config_name'])
            config.vocab_size = 50297
            self.model = AutoModel.from_pretrained(
                kwargs['model_name_or_path'],
                config=config
            )
            max_length = config.max_position_embeddings
            for idx in range(len(self.model.h)):
                self.model.h[idx].attn.bias = torch.ones(
                    (max_length, max_length), dtype=torch.bool
                ).view(1, 1, max_length, max_length)
        elif self.model_type == "codewhisperer_v2":
            self.tokenizer = AutoTokenizer.from_pretrained(kwargs['tokenizer_name'])
            self.tokenizer.add_special_tokens({'mask_token': '<mask>'})
            self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            config = AutoConfig.from_pretrained(kwargs['config_name'])
            config.vocab_size = 49154
            # print(self.tokenizer.vocab_size)
            # exit()
            self.model = AutoModel.from_pretrained(
                kwargs['model_name_or_path'],
                config=config
            )
            # print(self.model)
            # print(self.tokenizer)
            # exit()
            max_length = config.max_position_embeddings
            for idx in range(len(self.model.h)):
                self.model.h[idx].attn.bias = torch.ones(
                    (max_length, max_length), dtype=torch.bool
                ).view(1, 1, max_length, max_length)
        else:
            raise NotImplementedError

    def text_to_tensor(
            self,
            text: str,
            max_length: int = 1024,
            pad_to_max: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            max_length=max_length,
            pad_to_max_length=False,
            truncation=True
        )
        if pad_to_max and len(token_ids) < max_length:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (max_length - len(token_ids))
        if len(token_ids) > max_length:
            token_ids = token_ids[0:max_length]

        return torch.tensor(token_ids)

    def get_pad_id(self):
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor):
        return tokens_tensor != self.get_pad_id()

    def get_representations(self, list_input_ids, gpu_id=0):
        device = torch.device('cuda', gpu_id)
        self.model.to(device=device)
        self.model = self.model.to(dtype=torch.float16)
        self.model.eval()

        batch_size = 128
        sequence_outputs = []
        pooled_outputs = []
        attention_masks = []

        for idx in range(0, len(list_input_ids), batch_size):
            start, end = idx, min(idx + batch_size, len(list_input_ids))
            input_ids = torch.stack(list_input_ids[start:end], dim=0).to(device=device)
            attention_mask = self.get_attn_mask(input_ids)
            if self.model_type == "codewhisperer" or self.model_type == "codewhisperer_v2":
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                token_embeddings = output.hidden_states[-1]  # bsz x seq_len x hid_dim
            else:
                output = self.model(input_ids, attention_mask)
                token_embeddings = output.last_hidden_state  # bsz x seq_len x hid_dim

            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            sequence_embeddings = sum_embeddings / sum_mask  # bsz x hid_dim

            sequence_outputs.append(token_embeddings)
            pooled_outputs.append(sequence_embeddings)
            attention_masks.append(attention_mask)

        sequence_output = torch.cat(sequence_outputs)
        pooled_output = torch.cat(pooled_outputs)
        attention_mask = torch.cat(attention_masks)

        return sequence_output, pooled_output, attention_mask
    
    def compute_score(self, q_seq_rep, q_token_rep, c_seq_rep, c_token_rep, q_mask, c_mask):
        if self.sim_function == "pooled_sim":
            # compute sim based on pooled representations
            scores = torch.nn.functional.cosine_similarity(q_seq_rep, c_seq_rep).tolist()  # num_cand

        elif self.sim_function == "token_sim":
            # we compute similarity following ColBERT (https://arxiv.org/pdf/2004.12832.pdf)
            q_token_rep = torch.nn.functional.normalize(q_token_rep, dim=-1)
            c_token_rep = torch.nn.functional.normalize(c_token_rep, dim=-1)
            scores = torch.matmul(q_token_rep, c_token_rep.transpose(1, 2))  # num_cand x num_q_tok x num_c_tok
            scores = torch.max(scores, dim=-1)[0]  # num_cand x num_q_tok
            masked_scores = scores * q_mask  # ignoring pad tokens in relevance computation
            scores_sum = torch.sum(masked_scores, dim=1)
            num_non_zero = torch.sum(q_mask, dim=1)
            scores = (scores_sum / num_non_zero.float()).tolist()

        else:
            raise ValueError(f"unknown sim_function: {self.sim_function}")

        return scores

    def rerank(self, query: str, docs: List[str], doc_ids: List[str] = None, gpu_id=0):
        with torch.no_grad():
            batch_queries = [self.text_to_tensor(query, max_length=256)]
            batch_candidates = [self.text_to_tensor(d, max_length=256) for d in docs]

            # q_token_rep: num_q_tok x hidden_size, q_seq_rep: 1 x hidden_size
            q_token_rep, q_seq_rep, q_mask = self.get_representations(batch_queries, gpu_id)  # 1 x hidden_size
            # c_token_rep: num_c_tok x num_tok x hidden_size, c_seq_rep: num_cand x hidden_size
            c_token_rep, c_seq_rep, c_mask = self.get_representations(batch_candidates, gpu_id)  # num_cand x hidden_size

            scores = self.compute_score(q_seq_rep, q_token_rep, c_seq_rep, c_token_rep, q_mask, c_mask)

        # sorting
        if doc_ids is not None:
            doc_ids = [x for _, x in sorted(zip(scores, doc_ids), reverse=True)]
        docs_scores = [(x, s) for s, x in sorted(zip(scores, docs), reverse=True)]
        docs = [item[0] for item in docs_scores]
        scores = [item[1] for item in docs_scores]

        return docs, doc_ids, scores
