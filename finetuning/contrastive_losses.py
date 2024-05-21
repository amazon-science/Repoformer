import torch
import torch.nn.functional as F
from torch import nn


class ContraCLMTokLoss(nn.Module):
    def __init__(self, pad_token_id, temperature=0.05):
        super(ContraCLMTokLoss, self).__init__()
        self.temperature = temperature
        self.pad_token_id = pad_token_id
        print(f"Dropout-CL\t temperature: {temperature}")

    def contrastive_loss_token(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.size(0)

        features_1, features_2 = F.normalize(features_1, dim=1), F.normalize(features_2, dim=1)
        features = torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1*features_2, dim=-1).to(torch.float32) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()).to(torch.float32) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)

        Ng = neg.sum(dim=-1)
        loss = (- torch.log(pos / (Ng+pos))).mean()

        return loss

    def forward(self, last_hidden_states_1, last_hidden_states_2, token_mask_batch):
        '''
            last_hidden_states: bsz x seqlen x embed_dim
            logits: bsz x seqlen x vocab_size
            input_ids: bsz x seqlen
            labels: bsz x seqlen
        '''
        # Token-level Contrastive Loss w/ In-Sequence Negatives
        token_cl_loss = 0
        for idx, (token_embd_1, token_embd_2, token_mask) in enumerate(
                zip(last_hidden_states_1, last_hidden_states_2, token_mask_batch)):
            effect_token_1 = token_embd_1.masked_select(token_mask.unsqueeze(-1)).view(torch.sum(token_mask), -1)
            effect_token_2 = token_embd_2.masked_select(token_mask.unsqueeze(-1)).view(torch.sum(token_mask), -1)
            # print(f"token_mask_batch: {token_mask_batch.size()}/{token_mask_batch.sum(1)}")

            cl_loss = self.contrastive_loss_token(effect_token_1, effect_token_2)
            token_cl_loss += cl_loss
        token_cl_loss = token_cl_loss / len(last_hidden_states_1)
        return token_cl_loss
    

class ContraCLMSeqLoss(nn.Module):
    def __init__(self, pad_token_id, temperature=0.05):
        super(ContraCLMSeqLoss, self).__init__()
        self.pad_token_id = pad_token_id
        self.temperature = temperature
        print(f"Sequence-Level Contrastive Loss:\t temperature: {temperature}")

    def forward(self, last_hidden_states_1, last_hidden_states_2, token_mask):
        device = last_hidden_states_1.device
        batch_size = last_hidden_states_1.size(0)

        # get the sequence representation via mean pooling
        token_mask = token_mask.unsqueeze(-1)
        features_1 = torch.sum(last_hidden_states_1 * token_mask, dim=1) / torch.sum(token_mask, dim=1)
        features_2 = torch.sum(last_hidden_states_2 * token_mask, dim=1) / torch.sum(token_mask, dim=1)
        features_1, features_2 = F.normalize(features_1, dim=1), F.normalize(features_2, dim=1)
        features = torch.cat([features_1, features_2], dim=0)

        # create block diagonal mask to avoid contrast within the neighborhood of each example
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1 * features_2, dim=-1).to(torch.float32) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()).to(torch.float32) / self.temperature)
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        Ng = neg.sum(dim=-1)
        loss = (- torch.log(pos / (Ng + pos))).mean()

        return loss
