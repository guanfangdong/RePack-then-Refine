from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from repack.util import instantiate_from_config


class WrappedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()
    return np.random.seed(np.random.get_state()[1][0] + worker_info.id)


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_val_dataloader=False,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.wrap = wrap
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

    def prepare_data(self):
        for cfg in self.dataset_configs.values():
            instantiate_from_config(cfg)

    def setup(self, stage=None):
        self.datasets = {
            split: instantiate_from_config(cfg)
            for split, cfg in self.dataset_configs.items()
        }
        if self.wrap:
            self.datasets = {
                split: WrappedDataset(dataset)
                for split, dataset in self.datasets.items()
            }

    def _loader(self, split, shuffle=False):
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            worker_init_fn=init_fn,
            pin_memory=True,
            drop_last=(split == "train"),
        )

    def _train_dataloader(self):
        return self._loader("train", shuffle=True)

    def _val_dataloader(self, shuffle=False):
        return self._loader("validation", shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        return self._loader("test", shuffle=shuffle)

    def _predict_dataloader(self):
        return self._loader("predict", shuffle=False)
