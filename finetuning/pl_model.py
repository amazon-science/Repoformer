import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from deepspeed.ops.adam import FusedAdam
from transformers.optimization import get_linear_schedule_with_warmup, get_inverse_sqrt_schedule
from transformers.trainer_pt_utils import get_parameter_names

from utils import (get_inputs_and_labels,
                   get_inputs_and_labels_cfcinrc,
                   get_inputs_and_labels_separate_cfc_label, 
                   get_inputs_and_labels_separate_cfc_label_cfcinrc, 
                   load_model_and_tokenizer)


class LitContraCLM(pl.LightningModule):
    def __init__(self, trainer_args, loss_func_tok=None, loss_func_seq=None, 
                 loss_func_tok_word=None, num_nodes=1):
        super(LitContraCLM, self).__init__()
        self.save_hyperparameters(trainer_args)
        # Load Model and Tokenizer
        self.model, self.tokenizer = load_model_and_tokenizer(
            trainer_args.model_name, 
            pad_token_id=trainer_args.pad_token_id,
            dropout_layers=trainer_args.dropout_layers,
            dropout_p=trainer_args.dropout_p,
            functional_dropout=trainer_args.functional_dropout,
            add_repoformer_special_token=(trainer_args.loss=="Repoformer") and not trainer_args.debug_disable_adding_new_token,
            repoformer_cfc_in_rc=trainer_args.cfc_in_rc
        )
        self.trainer_args = trainer_args
        self.loss_func_tok = loss_func_tok
        self.loss_func_seq = loss_func_seq
        self.mle_loss = torch.nn.CrossEntropyLoss()
        self.vocab_size = self.model.config.vocab_size
        self.embed_dim = self.model.config.hidden_size
        self.num_nodes = num_nodes


    def setup(self, stage):
        if stage == 'fit':
            # Hyperparamters and Configuration
            self.dropout_p = self.trainer_args.dropout_p
            self.functional_dropout = self.trainer_args.functional_dropout
            self.pad_token_id = self.trainer_args.pad_token_id

            self.lr = self.trainer_args.lr
            self.weight_decay = self.trainer_args.weight_decay
            self.num_warmup_steps = self.trainer_args.warmup_steps
            self.num_epochs = self.trainer_args.max_epochs
            self.train_batch_size = self.trainer_args.train_batch_size
            self.num_train_examples = self.trainer_args.num_training_examples
            self.num_gpu_per_node = self.trainer_args.devices
            self.accumulate_grad_batches = self.trainer_args.accumulate_grad_batches

            if self.trainer_args.max_steps == -1:
                num_steps_per_epoch = self.num_train_examples // (self.num_gpu_per_node * self.num_nodes * self.accumulate_grad_batches)
                self.num_training_steps = self.num_epochs * num_steps_per_epoch
                print(f"steps_per_epoch: {num_steps_per_epoch}\t total_training_steps: {self.num_training_steps}.")
            else:
                self.num_training_steps = self.trainer_args.max_steps

            self.no_scheduling = self.trainer_args.no_scheduling
            self.inv_sqrt_scheduling = self.trainer_args.inv_sqrt_scheduling
            self.world_size = self.trainer_args.devices * self.num_nodes
            # Loss Configuration
            self.loss = self.trainer_args.loss
            assert self.loss in ["MLE_Only", "ContraCLM", "ContraCLMTok", "ContraCLMSeq", "Repoformer"], \
                f"Loss: `{self.loss}` is not supported!"


    def forward(self, input_ids, attention_mask=None):
        bsz, seqlen = input_ids.size()
        outputs = self.model(input_ids=input_ids, output_hidden_states=True)
        logits = outputs.logits
        assert logits.size() == torch.Size([bsz, seqlen, self.vocab_size])
        return logits, outputs.hidden_states


    def training_step(self, batch, batch_idx):
        token_ids = batch['input_ids']
        if self.loss == "Repoformer":
            if self.trainer_args.separate_cfc_token_loss:
                if self.trainer_args.cfc_in_rc:
                    input_ids, labels_cfc, labels_completion, attention_mask = get_inputs_and_labels_separate_cfc_label_cfcinrc(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_end_rc_token=self.tokenizer.vocab['<end_rc>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                        has_neg_retrieval=self.trainer_args.has_neg_retrieval
                    )
                else:
                    if self.trainer_args.has_neg_retrieval:
                        raise NotImplementedError
                    input_ids, labels_cfc, labels_completion, attention_mask = get_inputs_and_labels_separate_cfc_label(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_cfc_info_end_token=self.tokenizer.vocab['</cfc_info>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                        replace_cfc_end_with_fim_middle=self.trainer_args.replace_cfc_end_with_fim_middle
                    )
            else:
                if self.trainer_args.cfc_in_rc:
                    if self.trainer_args.has_neg_retrieval:
                        raise NotImplementedError
                    input_ids, labels, attention_mask = get_inputs_and_labels_cfcinrc(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_end_rc_token=self.tokenizer.vocab['<end_rc>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                    )
                else:
                    if self.trainer_args.has_neg_retrieval:
                        raise NotImplementedError
                    input_ids, labels, attention_mask = get_inputs_and_labels(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_cfc_info_end_token=self.tokenizer.vocab['</cfc_info>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                        replace_cfc_end_with_fim_middle=self.trainer_args.replace_cfc_end_with_fim_middle
                    )
        else:
            input_ids, labels, attention_mask = get_inputs_and_labels(
                token_ids, pad_token_id=self.pad_token_id, mask_pad=True
            )
        uniq_tokens = torch.unique(input_ids)
        all_tokens = torch.sum(attention_mask)
        self.log("all_tokens_per_gpu", all_tokens, sync_dist=True)
        self.log("unique_tokens_per_gpu", len(uniq_tokens), sync_dist=True)

        # first forward pass
        logits, hidden_states = self(input_ids, attention_mask=attention_mask)
        last_hidden_states = hidden_states[-1]

        # compute the MLE loss on all devices independently
        if self.trainer_args.separate_cfc_token_loss:
            assert self.trainer_args.cfc_token_loss_lambda is not None
            loss_cfc = self.mle_loss(logits.view(-1, self.vocab_size), labels_cfc.view(-1))
            # for batch with full negative retrieval instances, the loss_cfc may be nan.
            if torch.isnan(loss_cfc):
                loss_cfc = 0.0
            loss_code_completion = self.mle_loss(logits.view(-1, self.vocab_size), labels_completion.view(-1))
            loss = self.trainer_args.cfc_token_loss_lambda * loss_cfc + loss_code_completion
            # TODO: also log cfc accuracy/f1
            self.log("Train/Loss/MLE_cfc", loss_cfc, sync_dist=True, on_step=True)
            self.log("Train/Loss/MLE_code", loss_code_completion, sync_dist=True, on_step=True)
            self.log("Train/Loss/MLE", loss, sync_dist=True, on_step=True, prog_bar=True)
        else:
            loss = self.mle_loss(logits.view(-1, self.vocab_size), labels.view(-1))
            self.log("Train/Loss/MLE", loss, sync_dist=True, on_step=True, prog_bar=True)

        # Original MLE
        if self.loss == "MLE_Only" or self.loss == "Repoformer":
            return loss

        # get the dropout based augmentation either via the second forwarding pass or functional dropout
        if self.functional_dropout:
            last_hidden_states_orig = last_hidden_states
            last_hidden_states = F.dropout(last_hidden_states_orig, p=self.dropout_p)
            last_hidden_states_2 = F.dropout(last_hidden_states_orig, p=self.dropout_p)
        else:
            _, hidden_states_2 = self(input_ids, attention_mask=attention_mask)
            last_hidden_states_2 = hidden_states_2[-1]

        # Token-level loss
        if self.loss == "ContraCLMTok" or self.loss == "ContraCLM":
            loss_tok = self.loss_func_tok(last_hidden_states, last_hidden_states_2, attention_mask)
            loss += loss_tok
            self.log(f"Train/Loss/TokCL", loss_tok, sync_dist=True, on_step=True, prog_bar=True)

        # Sequence-level loss
        if self.loss == "ContraCLMSeq" or self.loss == "ContraCLM":
            # We use all_gather to gather representations from all GPUs. Since all_gather results are not part of
            # computational graph, we replace the current process's corresponding embeddings with original tensors
            if self.world_size > 1:
                all_attention_mask = self.all_gather(attention_mask).flatten(start_dim=0, end_dim=1)
                all_hidden_feature_1 = self.all_gather(last_hidden_states)
                all_hidden_feature_1[self.global_rank] = last_hidden_states
                all_hidden_feature_1 = all_hidden_feature_1.flatten(start_dim=0, end_dim=1)

                all_hidden_feature_2 = self.all_gather(last_hidden_states_2)
                all_hidden_feature_2[self.global_rank] = last_hidden_states_2
                all_hidden_feature_2 = all_hidden_feature_2.flatten(start_dim=0, end_dim=1)
            else:
                all_attention_mask = input_ids
                all_hidden_feature_1 = last_hidden_states
                all_hidden_feature_2 = last_hidden_states_2
            loss_seq = self.loss_func_seq(all_hidden_feature_1, all_hidden_feature_2, 
                                          all_attention_mask)
            loss += loss_seq
            self.log(f"Train/Loss/SeqCL", loss_seq, rank_zero_only=True, on_step=True, prog_bar=True)

        return loss


    def get_cfc_precision_recall(self, logits, labels, cfc_info_id, fim_middle_id, end_rc_id):
        has_cfc, pred_cfc = set(), set()

        cfc_rank_list = []
        for i in range(logits.shape[0]):
            label_list = labels[i].tolist()

            # skip cases with empty labels 
            # if self.trainer_args.has_neg_retrieval:
            if len((labels[i] != -100).nonzero(as_tuple=True)[0]) == 0:
                continue

            # recall
            cur_case_positive = False
            if cfc_info_id in label_list:
                # print('label has cfc')
                has_cfc.add(i)
                cur_case_positive = True
                
            # precision
            if fim_middle_id is not None:
                assert end_rc_id is None
                if fim_middle_id in label_list:
                    cfc_info_idx = (labels[i] == fim_middle_id).nonzero(as_tuple=True)[0].item() + 1
                else:
                    cfc_info_idx = (labels[i] != -100).nonzero(as_tuple=True)[0][0].item()
                if logits[i][cfc_info_idx].argmax().item() == cfc_info_id:
                    pred_cfc.add(i)
                    # print('pred has cfc')
                if cur_case_positive:
                    cfc_rank_list.append((logits[i][cfc_info_idx] > logits[i][cfc_info_idx][cfc_info_id]).sum().item() + 1)
            elif end_rc_id is not None:
                assert fim_middle_id is None
                if end_rc_id in label_list:
                    cfc_info_idx = (labels[i] == end_rc_id).nonzero(as_tuple=True)[0].item() + 1
                else:
                    cfc_info_idx = (labels[i] != -100).nonzero(as_tuple=True)[0][0].item()
                if logits[i][cfc_info_idx].argmax().item() == cfc_info_id:
                    pred_cfc.add(i)
                    # print('pred has cfc')
                if cur_case_positive:
                    cfc_rank_list.append((logits[i][cfc_info_idx] > logits[i][cfc_info_idx][cfc_info_id]).sum().item() + 1)
            else:
                raise NotImplementedError

        precision = len(pred_cfc.intersection(has_cfc)) / len(pred_cfc) if len(pred_cfc) > 0 else 0.0
        recall = len(pred_cfc.intersection(has_cfc)) / len(has_cfc) if len(has_cfc) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0.0

        mrr_list = [1/x for x in cfc_rank_list]
        mrr = sum(mrr_list) / len(mrr_list) if len(mrr_list) > 0 else 0.0

        return precision, recall, f1, mrr
    

    def validation_step(self, batch, batch_idx):
        if self.loss == "Repoformer":
            eval_items = []

            if self.trainer_args.separate_cfc_token_loss:
                eval_fct = torch.nn.CrossEntropyLoss()
                token_ids = batch['token_ids']
                if self.trainer_args.cfc_in_rc:
                    input_ids, labels_cfc, labels_completion, attention_mask = get_inputs_and_labels_separate_cfc_label_cfcinrc(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_end_rc_token=self.tokenizer.vocab['<end_rc>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                        has_neg_retrieval=self.trainer_args.has_neg_retrieval
                    )
                else:
                    if self.trainer_args.has_neg_retrieval:
                        raise NotImplementedError
                    input_ids, labels_cfc, labels_completion, attention_mask = get_inputs_and_labels_separate_cfc_label(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_cfc_info_end_token=self.tokenizer.vocab['</cfc_info>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                        replace_cfc_end_with_fim_middle=self.trainer_args.replace_cfc_end_with_fim_middle
                    )
                logits, _ = self(input_ids, attention_mask=attention_mask)
                loss_cfc = eval_fct(logits.view(-1, self.model.config.vocab_size), labels_cfc.view(-1))
                loss_code_completion = eval_fct(logits.view(-1, self.model.config.vocab_size), labels_completion.view(-1))
                loss = self.trainer_args.cfc_token_loss_lambda * loss_cfc + loss_code_completion
                eval_items += [loss_cfc, loss_code_completion, loss]
            else:
                eval_fct = torch.nn.CrossEntropyLoss()
                token_ids = batch['token_ids']
                if self.trainer_args.cfc_in_rc:
                    if self.trainer_args.has_neg_retrieval:
                        raise NotImplementedError
                    input_ids, labels, attention_mask = get_inputs_and_labels_cfcinrc(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_end_rc_token=self.tokenizer.vocab['<end_rc>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                    )
                else:
                    if self.trainer_args.has_neg_retrieval:
                        raise NotImplementedError
                    input_ids, labels, attention_mask = get_inputs_and_labels(
                        token_ids, pad_token_id=self.pad_token_id, mask_pad=True, 
                        repoformer_cfc_info_start_token=self.tokenizer.vocab['<cfc_info>'], 
                        repoformer_cfc_info_end_token=self.tokenizer.vocab['</cfc_info>'],
                        fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                        full_sequence_code_completion_loss=self.trainer_args.full_sequence_code_completion_loss,
                        replace_cfc_end_with_fim_middle=self.trainer_args.replace_cfc_end_with_fim_middle
                    )
                logits, _ = self(input_ids, attention_mask=attention_mask)
                loss = eval_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))
                eval_items.append(loss)

            # evaluate cfc predictions
            if self.trainer_args.separate_cfc_token_loss:
                labels = labels_cfc
            if self.trainer_args.valid_with_cfc_f1:
                if self.trainer_args.cfc_in_rc:
                    cfc_p, cfc_r, cfc_f1, cfc_mrr = self.get_cfc_precision_recall(logits, labels, 
                                                                                  cfc_info_id=self.tokenizer.vocab['<cfc_info>'],
                                                                                  fim_middle_id=None,
                                                                                  end_rc_id=self.tokenizer.vocab['<end_rc>'])
                else:
                    cfc_p, cfc_r, cfc_f1, cfc_mrr = self.get_cfc_precision_recall(logits, labels, 
                                                                                  cfc_info_id=self.tokenizer.vocab['<cfc_info>'],
                                                                                  fim_middle_id=self.tokenizer.vocab['<fim_middle>'],
                                                                                  end_rc_id=None)
                cfc_p = torch.tensor(cfc_p, dtype=loss.dtype, device=loss.device)
                cfc_r = torch.tensor(cfc_r, dtype=loss.dtype, device=loss.device)
                cfc_f1 = torch.tensor(cfc_f1, dtype=loss.dtype, device=loss.device)
                cfc_mrr = torch.tensor(cfc_mrr, dtype=loss.dtype, device=loss.device)
                eval_items += [cfc_p, cfc_r, cfc_f1, cfc_mrr]

            return eval_items
        
        else:
            eval_fct = torch.nn.CrossEntropyLoss()
            token_ids = batch['token_ids']
            input_ids, labels, attention_mask = get_inputs_and_labels(
                token_ids, pad_token_id=self.pad_token_id, mask_pad=True
            )
            logits, _ = self(input_ids, attention_mask=attention_mask)
            loss = eval_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

            return loss


    def validation_epoch_end(self, validation_step_outputs):
        if self.loss == "Repoformer":
            if self.trainer_args.separate_cfc_token_loss:
                loss_cfc = [x[0] for x in validation_step_outputs]
                loss_code_completion = [x[1] for x in validation_step_outputs]
                loss = [x[2] for x in validation_step_outputs]
                val_loss_cfc = torch.stack(loss_cfc).mean()
                val_loss_code_completion = torch.stack(loss_code_completion).mean()
                perplexity = torch.exp(val_loss_code_completion)
                val_loss = torch.stack(loss).mean()
                self.log("Valid/Loss/MLE_cfc", val_loss_cfc, sync_dist=True, on_epoch=True, prog_bar=False)
                self.log("Valid/Loss/MLE_code", val_loss_code_completion, sync_dist=True, on_epoch=True, prog_bar=True)
                self.log("Valid/Loss/Perplexity_code", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)
                self.log("Valid/Loss/MLE", val_loss, sync_dist=True, on_epoch=True, prog_bar=False)
            else:
                loss = [x[0] for x in validation_step_outputs]
                val_loss = torch.stack(loss).mean()
                perplexity = torch.exp(val_loss)
                self.log("Valid/Loss/MLE", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)
                self.log("Valid/Loss/Perplexity", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)
            
            if self.trainer_args.valid_with_cfc_f1:
                # assuming cfc_p, r, f1 are always the last three in items returned by valid step
                cfc_p = [x[-4] for x in validation_step_outputs]
                cfc_r = [x[-3] for x in validation_step_outputs]
                cfc_f1 = [x[-2] for x in validation_step_outputs]
                cfc_mrr = [x[-1] for x in validation_step_outputs]
                cfc_precision = torch.stack(cfc_p).mean()
                cfc_recall = torch.stack(cfc_r).mean()
                cfc_f1 = torch.stack(cfc_f1).mean()
                cfc_mrr = torch.stack(cfc_mrr).mean()
                self.log("Valid/CFC/Precision", cfc_precision, sync_dist=True, on_epoch=True, prog_bar=False)
                self.log("Valid/CFC/Recall", cfc_recall, sync_dist=True, on_epoch=True, prog_bar=False)
                self.log("Valid/CFC/F1", cfc_f1, sync_dist=True, on_epoch=True, prog_bar=True)
                self.log("Valid/CFC/MRR", cfc_mrr, sync_dist=True, on_epoch=True, prog_bar=True)
        else:
            val_loss = torch.stack(validation_step_outputs).mean()
            perplexity = torch.exp(val_loss)
            self.log("Valid/Loss/MLE", val_loss, sync_dist=True, on_epoch=True, prog_bar=True)
            self.log("Valid/Loss/Perplexity", perplexity, sync_dist=True, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optim_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = FusedAdam(optim_groups, lr=self.lr)
        # optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.no_scheduling:
            return optimizer
        if self.inv_sqrt_scheduling:
            scheduler = get_inverse_sqrt_schedule(optimizer, num_warmup_steps=self.num_warmup_steps)
        else:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.num_warmup_steps,
                                                        num_training_steps=self.num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]
