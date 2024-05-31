import torch
from torch.utils.data import Dataset

class LLMDataset(Dataset):
    def __init__(self, data):
        super(LLMDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        # indexing the chunked data directly
        source_tokens = torch.tensor(self.data[ind]['token_ids'])
        return {"input_ids": source_tokens}
