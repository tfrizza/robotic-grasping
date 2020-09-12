import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import glob
import os
import numpy as np
import logging
from argparse import ArgumentParser

from inference.post_process import post_process_output
from utils.dataset_processing import evaluation
from utils.data import get_dataset
from inference.models.grasp_model import ResidualBlock


class GraspModule(pl.LightningModule):

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--use-depth', type=int, default=1,
                            help='Use Depth image for training (1/0)')
        parser.add_argument('--use-rgb', type=int, default=1,
                            help='Use RGB image for training (1/0)')
        parser.add_argument('--use-dropout', type=int, default=1,
                            help='Use dropout for training (1/0)')
        parser.add_argument('--dropout-prob', type=float, default=0.1,
                            help='Dropout prob for training (0-1)')
        parser.add_argument('--channel-size', type=int, default=32,
                            help='Internal channel size for the network')

        parser.add_argument('--batch-size', type=int, default=8,
                            help='Batch size')
        parser.add_argument('--epochs', type=int, default=30,
                            help='Training epochs')

        parser.add_argument('--dataset', type=str,
                            help='Dataset Name ("cornell" or "jaquard")')
        parser.add_argument('--dataset-path', type=str,
                            help='Path to dataset')
        parser.add_argument('--split', type=float, default=0.9,
                            help='Fraction of data for training (remainder is validation)')
        parser.add_argument('--ds-shuffle', action='store_true', default=False,
                            help='Shuffle the dataset')
        parser.add_argument('--ds-rotate', type=float, default=0.0,
                            help='Shift the start point of the dataset to use a different test/train split')
        parser.add_argument('--random-seed', type=int, default=123,
                            help='Random seed for numpy')
        parser.add_argument('--description', type=str, default='',
                            help='Training description')
        return parser

    def __init__(self, hparams):
        super(GraspModule, self).__init__()
        self.hparams=hparams
        input_channels = 1 * self.hparams.use_depth + 3 * self.hparams.use_rgb
        output_channels = 1
        channel_size = self.hparams.channel_size
        dropout = self.hparams.use_dropout
        prob = self.hparams.dropout_prob

        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output

    def compute_loss(self, y_hat, y):
        y_pos, y_cos, y_sin, y_width = y
        pos_pred, cos_pred, sin_pred, width_pred = y_hat

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)

        return {
            'loss': p_loss + cos_loss + sin_loss + width_loss,
            'losses': {
                'p_loss': p_loss,
                'cos_loss': cos_loss,
                'sin_loss': sin_loss,
                'width_loss': width_loss
            },
            'pred': {
                'pos': pos_pred,
                'cos': cos_pred,
                'sin': sin_pred,
                'width': width_pred
            }
        }

    def predict(self, x):
        pos_pred, cos_pred, sin_pred, width_pred = self(x)
        return {
            'pos': pos_pred,
            'cos': cos_pred,
            'sin': sin_pred,
            'width': width_pred
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y, _, _, _ = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)

        result = pl.TrainResult(minimize=loss['loss'])
        logs = {'train_loss': loss['loss'],
                'train_p_loss': loss['losses']['p_loss'],
                'train_cos_loss': loss['losses']['cos_loss'],
                'train_sin_loss': loss['losses']['sin_loss'],
                'train_width_loss': loss['losses']['width_loss']
                }
        result.log_dict(logs)
        return result

    def validation_step(self, batch, batch_idx):
        x, y, didx, rot, zoom_factor = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)

        q_out, ang_out, w_out = post_process_output(loss['pred']['pos'], loss['pred']['cos'],
                                                    loss['pred']['sin'], loss['pred']['width'])

        s = evaluation.calculate_iou_match(q_out,
                                           ang_out,
                                           self.val_dataset.get_gtbb(didx, rot, zoom_factor),
                                           no_grasps=1,
                                           grasp_width=w_out,
                                           )

        result = pl.EvalResult()
        result.log('val_loss', loss['loss'], prog_bar=True)
        result.log_dict({'val_p_loss': loss['losses']['p_loss'],
                         'val_cos_loss': loss['losses']['cos_loss'],
                         'val_sin_loss': loss['losses']['sin_loss'],
                         'val_width_loss': loss['losses']['width_loss']
                         })
        result.log_dict({'correct': torch.tensor(float(s)),
                         'failed': torch.tensor(float(not s))},
                        on_epoch=True,
                        reduce_fx=torch.sum,
                        sync_dist=True,
                        sync_dist_op='sum')
        result.log('IoU', torch.tensor(float(s)),
                   prog_bar=True,
                   on_epoch=True,
                   reduce_fx=torch.mean,
                   sync_dist=True,
                   sync_dist_op='mean')
        return result

    def prepare_data(self):
        # download only
        pass

    def setup(self, stage):
        # transform
        if self.hparams.dataset == 'cornell':
            regex_pattern = 'pcd*cpos.txt'
        else:
            regex_pattern = '*/*_grasps.txt'
        grasp_files = glob.glob(os.path.join(self.hparams.dataset_path, '*', regex_pattern))
        indices = list(range(len(grasp_files)))
        split = int(np.floor(self.hparams.split * len(grasp_files)))
        if self.hparams.ds_shuffle:
            np.random.seed(self.hparams.random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[:split], indices[split:]
        train_files = [grasp_files[i] for i in train_indices]
        val_files = [grasp_files[i] for i in val_indices]
        print('Training size: {}'.format(len(train_indices)))
        print('Validation size: {}'.format(len(val_indices)))

        Dataset = get_dataset(self.hparams.dataset)
        grasp_train = Dataset(train_files,
                            ds_rotate=self.hparams.ds_rotate,
                            random_rotate=True,
                            random_zoom=True,
                            include_depth=self.hparams.use_depth,
                            include_rgb=self.hparams.use_rgb)
        grasp_val = Dataset(val_files,
                              ds_rotate=self.hparams.ds_rotate,
                              random_rotate=False,
                              random_zoom=False,
                              include_depth=self.hparams.use_depth,
                              include_rgb=self.hparams.use_rgb)

        # assign to use in dataloaders
        self.train_dataset = grasp_train
        self.val_dataset = grasp_val

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)
