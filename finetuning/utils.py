import logging
import os
import time

import GPUtil
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          GPT2LMHeadModel)

from contrastive_losses import ContraCLMSeqLoss, ContraCLMTokLoss

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def setup_log_path(args, num_nodes=1):
    respath = args.expt_prefix
    respath += f"_{args.loss}"
    respath += f"_bs{args.train_batch_size}"
    respath += f"_lr{args.lr}"
    if args.max_steps != -1:
        respath += f"_steps{args.max_steps}"
    else:
        num_steps_per_epoch = args.num_training_examples // (args.devices * num_nodes * args.accumulate_grad_batches)
        max_steps = args.max_epochs * num_steps_per_epoch
        respath += f"_steps{max_steps}"

    if args.model_name == "Salesforce/codegen-350M-mono":
        if args.functional_dropout:
            respath += f"_functional_dropout_{args.dropout_p}"
        elif args.dropout_layers != 0:
            respath += f"_dropout_{args.dropout_layers}_{args.dropout_p}"
    else:
        respath += f"_dropout_rate_{args.dropout_p}"
    respath += f"_warmup{args.warmup_steps}"
    respath += f"_wd{int(args.weight_decay * 100)}"
    respath += f"_temp{int(args.temperature*100)}"
    return respath


def load_model_and_tokenizer(model_name, pad_token_id, load_pretrained=True, dropout_layers=-1, dropout_p=0.1, 
                             functional_dropout=False, add_repoformer_special_token=False,
                             repoformer_cfc_in_rc=False):
    if model_name == "Salesforce/codegen-350M-mono":
        logger.info('Loading CodeGen model and tokenizer...')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = pad_token_id
        mod_config = AutoConfig.from_pretrained(model_name)

        # original CodeGen model does not have dropout
        if functional_dropout and dropout_layers != 0:
            raise ValueError("Default dropout and functional dropout cannot be applied simultaneously!")

        if dropout_layers == -1: # apply dropout to the whole model
            mod_config.attn_pdrop = dropout_p
            mod_config.embd_pdrop = dropout_p
            mod_config.resid_pdrop = dropout_p
        if load_pretrained:
            model = AutoModelForCausalLM.from_pretrained(model_name, config=mod_config)#, torch_dtype=torch.bfloat16)
        else:
            model = AutoModelForCausalLM.from_config(mod_config)#, torch_dtype=torch.bfloat16)

        if dropout_layers > 0: # add dropout to specified layers
            for layer_num in range(-1, -dropout_layers - 1, -1):
                model.transformer.h[layer_num].attn.attn_dropout.p = dropout_p
                model.transformer.h[layer_num].attn.resid_dropout.p = dropout_p
                model.transformer.h[layer_num].mlp.dropout.p = dropout_p
        logger.info(f"Number of active dropout layers inside model: {dropout_layers}")
        logger.info("Done.")

    elif model_name in ["gpt2", "gpt2-large"]:
        logger.info(f"functional_dropout: {functional_dropout}")
        assert not functional_dropout, "functional dropout should not be applied to None-CodeGen models!"
        logger.info("Loading GPT2 model and tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.bos_token
        pad_token_id = tokenizer.convert_tokens_to_ids([tokenizer.bos_token])[0]

        # if not dropout_p == 0.1:
        if not dropout_p == -1:
            mod_config = AutoConfig.from_pretrained(model_name)
            mod_config.attn_pdrop = dropout_p
            mod_config.embd_pdrop = dropout_p
            mod_config.resid_pdrop = dropout_p
            model = GPT2LMHeadModel.from_pretrained(model_name, config=mod_config)
            logger.info(f"\n Loaded GPT2 with Dropout: {dropout_p} \n")

        logger.info(f"pad_token={tokenizer.pad_token}, pad_token_id={pad_token_id}")
        logger.info("Done.")

    elif "starcoder" in model_name or "repoformer" in model_name:
        logger.info("Loading {} model and tokenizer...".format(model_name))

        tokenizer = AutoTokenizer.from_pretrained(model_name, vocab_size=49154, use_auth_token=True)
        # tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.bos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.bos_token_id

        mod_config = AutoConfig.from_pretrained(model_name, use_auth_token=True)
        assert not dropout_p == -1
        mod_config.attn_pdrop = dropout_p
        mod_config.embd_pdrop = dropout_p
        mod_config.resid_pdrop = dropout_p

        if load_pretrained:
            # torch_dtype=torch.bfloat16, use_auth_token=True
            model = AutoModelForCausalLM.from_pretrained(model_name, config=mod_config)
        else:
            # torch_dtype=torch.bfloat16, use_auth_token=True
            model = AutoModelForCausalLM.from_config(mod_config)
        model.gradient_checkpointing_enable()

        # inject special token here
        if add_repoformer_special_token:
            if repoformer_cfc_in_rc:
                tokenizer.add_tokens(['<cfc_info>', '<end_rc>']) 
            else:
                tokenizer.add_tokens(['<cfc_info>', '</cfc_info>']) 
            model.resize_token_embeddings(len(tokenizer))
        assert model.config.vocab_size == len(tokenizer.vocab) == 49154
            
        logger.info(f"\n Loaded StarCoder with Dropout: {dropout_p} \n")

        logger.info(f"pad_token={tokenizer.pad_token}, pad_token_id={pad_token_id}")
        logger.info("Done.")

    else:
        raise Exception(f"{model_name} is not supported!")

    logger.info(f"The vocabulary size of {model_name} is {len(tokenizer)}")
    time.sleep(2)
    # model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def load_deepspeed_state_dict(state_dict_path):
    state_dict = torch.load(state_dict_path, map_location="cpu")
    state_dict_attribute = None
    key_prefix = None
    if "state_dict" in state_dict:
        state_dict_attribute = "state_dict"
        key_prefix = "model"
    elif "module" in state_dict:
        state_dict_attribute = "module"
        key_prefix = "module.model"
    if state_dict_attribute:
        print(f"using state dict attribute {state_dict_attribute!r}")
        state_dict = state_dict[state_dict_attribute]
    if key_prefix:
        print(f"using state dict key prefix {key_prefix!r}")
        unwrapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(key_prefix):
                new_key = key[len(key_prefix) + 1 :]
                unwrapped_state_dict[new_key] = value
    else:
        unwrapped_state_dict = state_dict
    return unwrapped_state_dict


def get_inputs_and_labels(token_ids, pad_token_id=None, mask_pad=False, fim_middle_id=None,
                          repoformer_cfc_info_start_token=None, repoformer_cfc_info_end_token=None,
                          full_sequence_code_completion_loss=False,
                          replace_cfc_end_with_fim_middle=False):
    """
    Utility function to convert list of token IDs to inputs and labels.
    If mask_pad is True, the padding token in labels is replaced with -100. Attention_mask is computed that indicates
       which tokens are not padding tokens token_ids (torch.Tensor): bs x sq_len
    """
    # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L71
    inp_tensor = token_ids[:, :-1].clone()
    lbl_tensor = token_ids[:, 1:].clone()
    if mask_pad:
        assert pad_token_id is not None and type(pad_token_id) == int, 'Need valid token ID to mask'
        # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L77
        lbl_tensor[lbl_tensor[:, :] == pad_token_id] = -100

        # for repoformer
        if repoformer_cfc_info_start_token is not None and repoformer_cfc_info_end_token is not None:
            assert fim_middle_id is not None

            if full_sequence_code_completion_loss:
                # For cfc cases, mask everything to the left of (including) repoformer_cfc_info_end_token
                cfc_end_pos_mask = lbl_tensor.eq(repoformer_cfc_info_end_token).float()
                mask = cfc_end_pos_mask.cumsum(dim=1)
                mask = mask.logical_and(1 - cfc_end_pos_mask).float()

                # For cfc cases, unmask everything to the left of (including) repoformer_cfc_info_start_token
                cfc_start_pos_mask = lbl_tensor.eq(repoformer_cfc_info_start_token).float()
                mask = mask.logical_or(cfc_start_pos_mask).float()
                mask = mask.logical_or(1 - cfc_start_pos_mask.cumsum(dim=1)).float()
            else:
                # for example   
                # >>> lbl_tensor
                # tensor([[ -100,  -100,  -100,    12,    13,     0, 49152,     1,     2,     3, 49153,    15,    16,    17],
                #         [ -100,    12,    13,     0, 49152,     1,     2,     3,     4, 49153,    15,    16,    17,    18],
                #         [   19,     0, 49152,     1,     2,     3,     4,     5,     6,     7, 49153,    20,    21,    22]])
                # >>> cfc_end_pos_mask = lbl_tensor.eq(49153).float()
                # >>> mask = cfc_end_pos_mask.cumsum(dim=1)
                # >>> mask
                # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1.],
                #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.]])
                # >>> mask = mask.logical_and(1 - cfc_end_pos_mask).float()
                # >>> mask
                # tensor([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.],
                #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                #         [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]])
                # >>> cfc_start_pos_mask = lbl_tensor.eq(49152).float()
                # >>> mask = mask.logical_or(cfc_start_pos_mask).float()
                # >>> mask
                # tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1.],
                #         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]])
                # >>> fim_middle_pos_mask = lbl_tensor.eq(0).float()
                # >>> mask = mask.logical_and(1 - fim_middle_pos_mask).float()
                # >>> mask
                # tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1.],
                #         [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1.],
                #         [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.]])

                # For cfc cases, mask everything to the left of (including) repoformer_cfc_info_end_token
                cfc_end_pos_mask = lbl_tensor.eq(repoformer_cfc_info_end_token).float()
                mask = cfc_end_pos_mask.cumsum(dim=1)
                mask = mask.logical_and(1 - cfc_end_pos_mask).float()

                # For cfc cases, unmask repoformer_cfc_info_start_token
                cfc_start_pos_mask = lbl_tensor.eq(repoformer_cfc_info_start_token).float()
                mask = mask.logical_or(cfc_start_pos_mask).float()
                
                # For cases without cfc, mask everything to the left of <fim_middle>
                no_cfc_mask = torch.logical_not(torch.any(lbl_tensor == repoformer_cfc_info_start_token, dim=1)).float()
                fim_middle_id_pos_mask = lbl_tensor.eq(fim_middle_id).float()
                mask_infile_cases = fim_middle_id_pos_mask.cumsum(dim=1)
                mask[no_cfc_mask == 1] = mask_infile_cases[no_cfc_mask == 1]

                # in addition, mask <fim_middle>
                fim_middle_pos_mask = lbl_tensor.eq(fim_middle_id).float()
                mask = mask.logical_and(1 - fim_middle_pos_mask).float()

            # For the pure CLM cases, do not mask anything
            pure_clm_mask = torch.logical_not(torch.any(lbl_tensor == fim_middle_id, dim=1)).float()
            mask[pure_clm_mask == 1] = 1

            # mask labels
            lbl_tensor[mask == 0] = -100

        # debugging
        '''
        lbl_tensor_temp = token_ids[:, 1:].clone()
        lbl_tensor_temp[lbl_tensor_temp[:, :] == pad_token_id] = -100
        labels_list = lbl_tensor_temp.squeeze().tolist()
        mask_list = mask.squeeze().tolist()
        cfc_info_start_idx = None 
        cfc_info_end_idx = None
        code_completion_lbl_tensor_final = lbl_tensor.squeeze().tolist()

        if 49152 in labels_list:
            cfc_info_start_idx = labels_list.index(49152)
            cfc_info_end_idx = labels_list.index(49153)
            labels_list[cfc_info_start_idx] = [[labels_list[cfc_info_start_idx]]]
            labels_list[cfc_info_end_idx] = [[labels_list[cfc_info_end_idx]]]
            mask_list[cfc_info_start_idx] = [[mask_list[cfc_info_start_idx]]]
            mask_list[cfc_info_end_idx] = [[mask_list[cfc_info_end_idx]]]
            code_completion_lbl_tensor_final[cfc_info_start_idx] = [[code_completion_lbl_tensor_final[cfc_info_start_idx]]]
            code_completion_lbl_tensor_final[cfc_info_end_idx] = [[code_completion_lbl_tensor_final[cfc_info_end_idx]]]
        print(labels_list)
        print('=======================')
        print(mask_list)
        print('=======================')
        print(code_completion_lbl_tensor_final)
        print('=======================')
        print()
        print('<fim_middle> at {}'.format(labels_list.index(2)))
        print('<cfc_info> at {}'.format(cfc_info_start_idx))
        print('</cfc_info> at {}'.format(cfc_info_end_idx))
        print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n', flush=True)
        # exit()
        '''

        attention_mask = torch.ones_like(inp_tensor)
        attention_mask = attention_mask.masked_fill(inp_tensor.eq(pad_token_id), 0.0).type(torch.bool)

    # additional processing on the input side
    if replace_cfc_end_with_fim_middle:
        inp_tensor[inp_tensor == repoformer_cfc_info_end_token] = fim_middle_id

    return inp_tensor, lbl_tensor, attention_mask


def get_inputs_and_labels_cfcinrc(token_ids, pad_token_id=None, mask_pad=False, fim_middle_id=None,
                                  repoformer_cfc_info_start_token=None,  repoformer_end_rc_token=None,
                                  full_sequence_code_completion_loss=False):
    """
    Utility function to convert list of token IDs to inputs and labels.
    If mask_pad is True, the padding token in labels is replaced with -100. Attention_mask is computed that indicates
       which tokens are not padding tokens token_ids (torch.Tensor): bs x sq_len
    """
    # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L71
    inp_tensor = token_ids[:, :-1].clone()
    lbl_tensor = token_ids[:, 1:].clone()
    if mask_pad:
        assert pad_token_id is not None and type(pad_token_id) == int, 'Need valid token ID to mask'
        # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L77
        lbl_tensor[lbl_tensor[:, :] == pad_token_id] = -100

        # for repoformer
        if repoformer_cfc_info_start_token is not None and repoformer_end_rc_token is not None:
            assert fim_middle_id is not None

            if full_sequence_code_completion_loss:
                # Mask everything to the left of <fim_middle>
                fim_middle_id_pos_mask = lbl_tensor.eq(fim_middle_id).float()
                mask = fim_middle_id_pos_mask.cumsum(dim=1)

                # in addition, mask <fim_middle>
                fim_middle_pos_mask = lbl_tensor.eq(fim_middle_id).float()
                mask = mask.logical_and(1 - fim_middle_pos_mask).float()

                # unmask repoformer_cfc_info_start_token
                cfc_start_pos_mask = lbl_tensor.eq(repoformer_cfc_info_start_token).float()
                mask = mask.logical_or(cfc_start_pos_mask).float()

                # Then, unmask everything to the left of end_rc
                endrc_start_pos_mask = lbl_tensor.eq(repoformer_end_rc_token).float()
                mask = mask.logical_or(repoformer_end_rc_token).float()   # unmask repoformer_cfc_info_start_token)
                mask = mask.logical_or(1 - endrc_start_pos_mask.cumsum(dim=1)).float()
                
            else:
                # Mask everything to the left of <fim_middle>
                fim_middle_id_pos_mask = lbl_tensor.eq(fim_middle_id).float()
                mask = fim_middle_id_pos_mask.cumsum(dim=1)

                # in addition, mask <fim_middle>
                fim_middle_pos_mask = lbl_tensor.eq(fim_middle_id).float()
                mask = mask.logical_and(1 - fim_middle_pos_mask).float()

                # unmask repoformer_cfc_info_start_token
                cfc_start_pos_mask = lbl_tensor.eq(repoformer_cfc_info_start_token).float()
                mask = mask.logical_or(cfc_start_pos_mask).float()

            # For the pure CLM cases, do not mask anything
            pure_clm_mask = torch.logical_not(torch.any(lbl_tensor == fim_middle_id, dim=1)).float()
            mask[pure_clm_mask == 1] = 1

            # mask labels
            lbl_tensor[mask == 0] = -100

        attention_mask = torch.ones_like(inp_tensor)
        attention_mask = attention_mask.masked_fill(inp_tensor.eq(pad_token_id), 0.0).type(torch.bool)

    return inp_tensor, lbl_tensor, attention_mask


def get_inputs_and_labels_separate_cfc_label(token_ids, pad_token_id=None, mask_pad=False, fim_middle_id=None,
                                             repoformer_cfc_info_start_token=None, repoformer_cfc_info_end_token=None,
                                             full_sequence_code_completion_loss=False,
                                             replace_cfc_end_with_fim_middle=False):
    """
    Utility function to convert list of token IDs to inputs and labels.
    If mask_pad is True, the padding token in labels is replaced with -100. Attention_mask is computed that indicates
       which tokens are not padding tokens token_ids (torch.Tensor): bs x sq_len
    """
    assert repoformer_cfc_info_start_token is not None and repoformer_cfc_info_end_token is not None
    # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L71
    inp_tensor = token_ids[:, :-1].clone()
    cfc_lbl_tensor = token_ids[:, 1:].clone()
    code_completion_lbl_tensor = token_ids[:, 1:].clone()
    if mask_pad:
        assert pad_token_id is not None and type(pad_token_id) == int, 'Need valid token ID to mask'
        # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L77
        code_completion_lbl_tensor[code_completion_lbl_tensor[:, :] == pad_token_id] = -100
        cfc_lbl_tensor[cfc_lbl_tensor[:, :] == pad_token_id] = -100

        # for repoformer
        if repoformer_cfc_info_start_token is not None and repoformer_cfc_info_end_token is not None:
            assert fim_middle_id is not None

            #####################
            # part 1 - mask
            #####################
            if full_sequence_code_completion_loss:
                
                # For cfc cases, mask everything to the left of (including) repoformer_cfc_info_end_token
                cfc_end_pos_mask = code_completion_lbl_tensor.eq(repoformer_cfc_info_end_token).float()
                mask = cfc_end_pos_mask.cumsum(dim=1)
                mask = mask.logical_and(1 - cfc_end_pos_mask).float()

                # For cfc cases, unmask everything to the left of (including) repoformer_cfc_info_start_token
                cfc_start_pos_mask = code_completion_lbl_tensor.eq(repoformer_cfc_info_start_token).float()
                # mask = mask.logical_or(cfc_start_pos_mask).float()   # (no need to unmask repoformer_cfc_info_start_token)
                mask = mask.logical_or(1 - cfc_start_pos_mask.cumsum(dim=1)).float()
                
                # mask labels
                code_completion_lbl_tensor[mask == 0] = -100

            else:
                # For cfc cases, mask everything to the left of (including) repoformer_cfc_info_end_token
                cfc_end_pos_mask = code_completion_lbl_tensor.eq(repoformer_cfc_info_end_token).float()
                mask = cfc_end_pos_mask.cumsum(dim=1)
                mask = mask.logical_and(1 - cfc_end_pos_mask).float()

                # (no need to unmask repoformer_cfc_info_start_token)
                # cfc_start_pos_mask = code_completion_lbl_tensor.eq(repoformer_cfc_info_start_token).float()
                # mask = mask.logical_or(cfc_start_pos_mask).float()
                
                # For cases without cfc, mask everything to the left of <fim_middle>
                no_cfc_mask = torch.logical_not(torch.any(code_completion_lbl_tensor == repoformer_cfc_info_start_token, dim=1)).float()
                fim_middle_id_pos_mask = code_completion_lbl_tensor.eq(fim_middle_id).float()
                mask_infile_cases = fim_middle_id_pos_mask.cumsum(dim=1)
                mask[no_cfc_mask == 1] = mask_infile_cases[no_cfc_mask == 1]

                # in addition, mask <fim_middle>
                fim_middle_pos_mask = code_completion_lbl_tensor.eq(fim_middle_id).float()
                mask = mask.logical_and(1 - fim_middle_pos_mask).float()

                # For the pure CLM cases, do not mask anything
                pure_clm_mask = torch.logical_not(torch.any(code_completion_lbl_tensor == fim_middle_id, dim=1)).float()
                mask[pure_clm_mask == 1] = 1

                # mask labels
                code_completion_lbl_tensor[mask == 0] = -100
                
            #####################
            # part 2 - mask_cfc (same for both full_sequence_code_completion_loss = True or False)
            #####################
            # Find the position of <fim_middle>
            # The position to keep is just one element to the right
            mask_cfc = cfc_lbl_tensor.eq(fim_middle_id).float()
            mask_cfc = torch.roll(mask_cfc, shifts=1, dims=-1)

            # mask labels
            cfc_lbl_tensor[mask_cfc == 0] = -100
            
        '''
        # debugging
        cfc_lbl_tensor_temp = token_ids[:, 1:].clone()
        cfc_lbl_tensor_temp[cfc_lbl_tensor_temp[:, :] == pad_token_id] = -100
        labels_list = cfc_lbl_tensor_temp.squeeze().tolist()
        mask_list = mask.squeeze().tolist()
        mask_cfc_list = mask_cfc.squeeze().tolist()
        cfc_info_start_idx = None 
        cfc_info_end_idx = None
        code_completion_lbl_tensor_final = code_completion_lbl_tensor.squeeze().tolist()
        cfc_lbl_tensor_final = cfc_lbl_tensor.squeeze().tolist()

        if 49152 in labels_list:
            cfc_info_start_idx = labels_list.index(49152)
            cfc_info_end_idx = labels_list.index(49153)
            labels_list[cfc_info_start_idx] = [[labels_list[cfc_info_start_idx]]]
            labels_list[cfc_info_end_idx] = [[labels_list[cfc_info_end_idx]]]
            mask_list[cfc_info_start_idx] = [[mask_list[cfc_info_start_idx]]]
            mask_list[cfc_info_end_idx] = [[mask_list[cfc_info_end_idx]]]
            mask_cfc_list[cfc_info_start_idx] = [[mask_cfc_list[cfc_info_start_idx]]]
            mask_cfc_list[cfc_info_end_idx] = [[mask_cfc_list[cfc_info_end_idx]]]
            code_completion_lbl_tensor_final[cfc_info_start_idx] = [[code_completion_lbl_tensor_final[cfc_info_start_idx]]]
            code_completion_lbl_tensor_final[cfc_info_end_idx] = [[code_completion_lbl_tensor_final[cfc_info_end_idx]]]
            cfc_lbl_tensor_final[cfc_info_start_idx] = [[cfc_lbl_tensor_final[cfc_info_start_idx]]]
            cfc_lbl_tensor_final[cfc_info_end_idx] = [[cfc_lbl_tensor_final[cfc_info_end_idx]]]
        print(labels_list)
        print('=======================')
        print(mask_list)
        # print(mask_cfc_list)
        print('=======================')
        print(code_completion_lbl_tensor_final)
        # print(cfc_lbl_tensor_final)
        print('=======================')
        print()
        print('<fim_middle> at {}'.format(labels_list.index(2)))
        print('<cfc_info> at {}'.format(cfc_info_start_idx))
        print('</cfc_info> at {}'.format(cfc_info_end_idx))
        print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n', flush=True)
        # exit()
        '''
        
        attention_mask = torch.ones_like(inp_tensor)
        attention_mask = attention_mask.masked_fill(inp_tensor.eq(pad_token_id), 0.0).type(torch.bool)
        
    # additional processing on the input side
    if replace_cfc_end_with_fim_middle:
        inp_tensor[inp_tensor == repoformer_cfc_info_end_token] = fim_middle_id

    return inp_tensor, cfc_lbl_tensor, code_completion_lbl_tensor, attention_mask


def get_inputs_and_labels_separate_cfc_label_cfcinrc(token_ids, pad_token_id=None, mask_pad=False, fim_middle_id=None,
                                                     repoformer_cfc_info_start_token=None, repoformer_end_rc_token=None,
                                                     full_sequence_code_completion_loss=False,
                                                     has_neg_retrieval=False):
    """
    Utility function to convert list of token IDs to inputs and labels.
    If mask_pad is True, the padding token in labels is replaced with -100. Attention_mask is computed that indicates
       which tokens are not padding tokens token_ids (torch.Tensor): bs x sq_len
    """
    assert repoformer_cfc_info_start_token is not None and repoformer_end_rc_token is not None
    # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L71
    inp_tensor = token_ids[:, :-1].clone()
    cfc_lbl_tensor = token_ids[:, 1:].clone()
    code_completion_lbl_tensor = token_ids[:, 1:].clone()
    if mask_pad:
        assert pad_token_id is not None and type(pad_token_id) == int, 'Need valid token ID to mask'
        # https://github.com/yxuansu/SimCTG/blob/4b2bf5b77c9bd54edc195e048c30a369e0f55ef4/training_tutorial_on_wikitext103/dataclass.py#L77
        code_completion_lbl_tensor[code_completion_lbl_tensor[:, :] == pad_token_id] = -100
        cfc_lbl_tensor[cfc_lbl_tensor[:, :] == pad_token_id] = -100

        # for repoformer
        if repoformer_cfc_info_start_token is not None and repoformer_end_rc_token is not None:
            assert fim_middle_id is not None

            #####################
            # part 1 - mask
            #####################
            if full_sequence_code_completion_loss:
                # Mask everything to the left of <fim_middle>
                fim_middle_id_pos_mask = code_completion_lbl_tensor.eq(fim_middle_id).float()
                mask = fim_middle_id_pos_mask.cumsum(dim=1)

                # in addition, mask <fim_middle>
                fim_middle_pos_mask = code_completion_lbl_tensor.eq(fim_middle_id).float()
                mask = mask.logical_and(1 - fim_middle_pos_mask).float()

                # Then, unmask everything to the left of end_rc
                endrc_start_pos_mask = code_completion_lbl_tensor.eq(repoformer_end_rc_token).float()
                # mask = mask.logical_or(cfc_start_pos_mask).float()   # (no need to unmask repoformer_cfc_info_start_token)
                # mask = mask.logical_or(endrc_start_pos_mask).float()   # (also, no need to unmask repoformer_end_rc_token)
                mask = mask.logical_or(1 - endrc_start_pos_mask.cumsum(dim=1)).float()
                
                # mask labels
                code_completion_lbl_tensor[mask == 0] = -100

            else:
                # (no need to unmask repoformer_cfc_info_start_token)
                # cfc_start_pos_mask = code_completion_lbl_tensor.eq(repoformer_cfc_info_start_token).float()
                # mask = mask.logical_or(cfc_start_pos_mask).float()
                
                # For all cases, mask everything to the left of <fim_middle>
                fim_middle_id_pos_mask = code_completion_lbl_tensor.eq(fim_middle_id).float()
                mask = fim_middle_id_pos_mask.cumsum(dim=1)

                # in addition, mask <fim_middle>
                fim_middle_pos_mask = code_completion_lbl_tensor.eq(fim_middle_id).float()
                mask = mask.logical_and(1 - fim_middle_pos_mask).float()

                # For the pure CLM cases, do not mask anything
                pure_clm_mask = torch.logical_not(torch.any(code_completion_lbl_tensor == fim_middle_id, dim=1)).float()
                mask[pure_clm_mask == 1] = 1

                # mask labels
                code_completion_lbl_tensor[mask == 0] = -100
            
            #####################
            # part 2 - mask_cfc (same for both full_sequence_code_completion_loss = True or False)
            #####################
            # Find the position of <end_rc>
            # The position to keep is just one element to the right
            mask_cfc = cfc_lbl_tensor.eq(repoformer_end_rc_token).float()
            mask_cfc = torch.roll(mask_cfc, shifts=1, dims=-1)

            # mask labels
            cfc_lbl_tensor[mask_cfc == 0] = -100

        # special processing for has_neg_retrieval
        if has_neg_retrieval:
            repoformer_neg_cfc_token = repoformer_end_rc_token + 1    # this is a special token for neg retrieval 
            cfc_lbl_tensor[cfc_lbl_tensor == repoformer_neg_cfc_token] = -100   # do not calculate any loss on <cfc_info> for negative retrievals
            inp_tensor[inp_tensor == repoformer_neg_cfc_token] = repoformer_cfc_info_start_token   # replace <neg_cfc_info> with <cfc_info> for code completion
        
        attention_mask = torch.ones_like(inp_tensor)
        attention_mask = attention_mask.masked_fill(inp_tensor.eq(pad_token_id), 0.0).type(torch.bool)
        
    return inp_tensor, cfc_lbl_tensor, code_completion_lbl_tensor, attention_mask


def get_loss_func(args, pad_token_id):
    """ get the contrastive learning loss function """
    logger.info(f"Getting {args.loss} objective")

    assert args.loss in ["MLE_Only", "ContraCLM", "ContraCLMTok", "ContraCLMSeq", "Repoformer"], \
        f"Loss: `{args.loss}` is not supported!"

    # get the token-level contrastive loss
    if args.loss == 'ContraCLMTok' or args.loss == 'ContraCLM':
        loss_func_tok = ContraCLMTokLoss(pad_token_id, args.temperature)
    else:
        loss_func_tok = None
    
    if args.loss == 'ContraCLMSeq' or args.loss == 'ContraCLM':
        loss_func_seq = ContraCLMSeqLoss(pad_token_id, args.temperature)
    else:
        loss_func_seq = None

    return loss_func_tok, loss_func_seq


class LitProgressBar(TQDMProgressBar):
    '''Overriding progress bar metrics to display more meaningful stats than defaults'''
    def __init__(self, total_train_steps, grad_accumulate_steps):
        super(LitProgressBar, self).__init__()
        self.total_train_steps = total_train_steps * grad_accumulate_steps

    def on_train_epoch_start(self, trainer, *_):
        total_train_batches = self.total_train_steps
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf") and total_val_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch
        total_batches = total_train_batches + total_val_batches
        self.main_progress_bar.reset(total_batches)

    def on_train_batch_end(self, trainer, pl_module, *_):
        def _update_n(bar, current, refresh_rate):
            if not bar.disable:
                total = bar.total
                leftover = current % refresh_rate
                advance = leftover if (current == total and leftover != 0) else refresh_rate
                bar.update(advance)
                bar.refresh()
        current = self.train_batch_idx + self._val_processed
        if self._should_update(current, self.main_progress_bar.total):
            _update_n(self.main_progress_bar, current, self.refresh_rate)
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
            
            
class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency=5000,
        prefix="NStep-ckpt",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if (global_step > 0) and global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

class GPUtilCallback(Callback):
    def __init__(self):
        self.batch = 0

    @rank_zero_only
    def on_train_batch_end(self, *args, **kwargs):
        # print gpu utils at the beginning of training at rank 0
        self.batch += 1
        if self.batch == 1 or self.batch == 100:
            GPUtil.showUtilization(all=True)
