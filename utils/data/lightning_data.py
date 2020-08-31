import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import random
import glob
import os
import numpy as np
import logging

# Note - you must have torchvision installed for this example
from torchvision import transforms

from utils.data import get_dataset


class GraspDataModule(pl.LightningDataModule):
    def __init__(self, args, batch_size=8):
        super().__init__()
        self.batch_size = batch_size
        self.args = args

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage):
        # transform
        grasp_files = glob.glob(os.path.join(self.args.dataset_path, '*', 'pcd*cpos.txt'))
        indices = list(range(len(grasp_files)))
        split = int(np.floor(self.args.split * len(grasp_files)))
        if self.args.ds_shuffle:
            np.random.seed(self.args.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        train_files = [grasp_files[i] for i in train_indices]
        val_files = [grasp_files[i] for i in val_indices]
        logging.info('Training size: {}'.format(len(train_indices)))
        logging.info('Validation size: {}'.format(len(val_indices)))

        Dataset = get_dataset(self.args.dataset)
        grasp_train = Dataset(train_files,
                            ds_rotate=self.args.ds_rotate,
                            random_rotate=True,
                            random_zoom=True,
                            include_depth=self.args.use_depth,
                            include_rgb=self.args.use_rgb)
        grasp_val = Dataset(val_files,
                              ds_rotate=self.args.ds_rotate,
                              random_rotate=False,
                              random_zoom=False,
                              include_depth=self.args.use_depth,
                              include_rgb=self.args.use_rgb)

        # assign to use in dataloaders
        self.train_dataset = grasp_train
        self.val_dataset = grasp_val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)
