# src/data/preprocessing/data_modules.py

import torch
import numpy as np
import os
from functools import partial
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

# class _RepeatSampler(object):
#     """ Sampler that repeats forever.

#     Args:
#         sampler (Sampler)
#     """

#     def __init__(self, sampler):
#         self.sampler = sampler
#         # Set instance variables dynamically
#         for attr_name in dir(sampler):
#             if not hasattr(self, attr_name):
#                 setattr(self, attr_name, getattr(sampler, attr_name))

#     def __iter__(self):
#         while True:
#             yield from iter(self.sampler)

# class MultiEpochsDataLoader(DataLoader):
#     """Custom DataLoader that repeats the dataset forever.
#     source: 
#     https://github.com/huggingface/pytorch-image-models/blob/d72ac0db259275233877be8c1d4872163954dfbb/timm/data/loader.py#L209-L238
#     So we can reuse worker processes after each epoch to reduce time spent on data loading.
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._DataLoader__initialized = False
#         self.batch_sampler = _RepeatSampler(self.batch_sampler)
#         self._DataLoader__initialized = True
#         self.iterator = super().__iter__()

#     def __len__(self):
#         return len(self.batch_sampler.sampler)

#     def __iter__(self):
#         for i in range(len(self)):
#             yield next(self.iterator)

def worker_init_fn(_):
    """Initialize the worker process."""
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataModuleFromConfig(pl.LightningDataModule):
    """DataModule from config. This is a simple wrapper around the LightningDataModule"""
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                     wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                     shuffle_val_dataloader=False, persistent_workers=True):
            """Initialize the DataModule.

            Args:
                batch_size (int): The batch size for the dataloaders.
                train (Dataset, optional): The training dataset. Defaults to None.
                validation (Dataset, optional): The validation dataset. Defaults to None.
                test (Dataset, optional): The test dataset. Defaults to None.
                predict (Dataset, optional): The prediction dataset. Defaults to None.
                wrap (bool, optional): Whether to wrap the dataloaders with additional functionality. Defaults to False.
                num_workers (int, optional): The number of worker processes to use for data loading. Defaults to None.
                shuffle_test_loader (bool, optional): Whether to shuffle the test dataloader. Defaults to False.
                use_worker_init_fn (bool, optional): Whether to use a worker_init_fn for data loading. Defaults to False.
                shuffle_val_dataloader (bool, optional): Whether to shuffle the validation dataloader. Defaults to False.
            """
            super().__init__()
            self.batch_size = batch_size
            self.dataset_configs = dict()
            self.persistent_workers = persistent_workers
            self.num_workers = min(num_workers, os.cpu_count()) if num_workers is not None else os.cpu_count()
            self.use_worker_init_fn = use_worker_init_fn
            if train is not None:
                self.dataset_configs["train"] = train
                self.train_dataloader = self._train_dataloader
            if validation is not None:
                self.dataset_configs["validation"] = validation
                self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
            if test is not None:
                self.dataset_configs["test"] = test
                self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
            if predict is not None:
                self.dataset_configs["predict"] = predict
                self.predict_dataloader = self._predict_dataloader
            self.wrap = wrap

    def prepare_data(self):
        """Prepare data, how to download, tokenize, etc."""
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        """Setup data, how to split, define dataset, etc."""
        
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        """generate the training dataloader(s)"""
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, pin_memory=torch.cuda.is_available())

    def _val_dataloader(self, shuffle=False):
        """generate the validation dataloader(s)"""
        
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=self.persistent_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle,
                          pin_memory=torch.cuda.is_available())

    def _test_dataloader(self, shuffle=False):
        """generate the test dataloader(s)"""
        
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers, worker_init_fn=init_fn, shuffle=shuffle, 
                          pin_memory=torch.cuda.is_available())

    def _predict_dataloader(self, shuffle=False):
        """generate the predict dataloader(s)"""
        
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, persistent_workers=self.persistent_workers, worker_init_fn=init_fn, 
                          pin_memory=torch.cuda.is_available())