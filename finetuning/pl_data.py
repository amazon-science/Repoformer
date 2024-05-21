import logging
import time

import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from dataloader.bigquery_pypi import LLMDataset


class RepoformerDataModule(pl.LightningDataModule):
    def __init__(self, data_prefix, train_datadir, valid_datadir, train_batch_size, 
                 valid_batch_size, num_workers=0):
        super(RepoformerDataModule, self).__init__()
        self.data_prefix = data_prefix
        self.train_datadir = train_datadir
        self.valid_datadir = valid_datadir
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers
        logger.info(f"Initializing RepoformerDataModule w/ train_bs={self.train_batch_size}, "
                    f"valid_bs={self.valid_batch_size}")

    def setup(self, stage=None):
        '''Called by every process'''
        logger.info('Loading data...')
        self.valid_data = load_from_disk(self.valid_datadir)
        self.valid_data.set_format(type='torch', columns=['token_ids'])

        # with original data
        if "wikitext" in self.data_prefix.lower():
            self.train_data = load_from_disk(self.train_datadir)
            self.train_data.set_format(type='torch', columns=['input_ids'])
        elif "bigquery" in self.data_prefix.lower() or 'repoformer' in self.data_prefix.lower():
            # load the original data
            train_orig_data = load_from_disk(self.train_datadir)
            self.train_data = LLMDataset(train_orig_data)
        else:
            raise ValueError(f"The {self.data_prefix} data is currently not supported!")

        logger.info(f'Loaded Train data with {len(self.train_data)} examples')
        logger.info(f"train_bs={self.train_batch_size}\t "
                    f"valid_bs={self.valid_batch_size}")
        time.sleep(5)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.train_batch_size, 
                          num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.valid_batch_size, 
                          num_workers=8, shuffle=False)

# References
# ----------
# https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
