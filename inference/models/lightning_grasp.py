import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import glob
import os
import numpy as np
import logging

from inference.post_process import post_process_output
from utils.dataset_processing import evaluation
from utils.data import get_dataset


class GraspModule(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, x_in):
        raise NotImplementedError()

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

        # (log keyword is optional)
        # result = pl.TrainResult(minimize=loss['loss'])
        # result.log('val_loss', loss['loss'], prog_bar=True)
        # logs = {'train_loss': loss['loss'],
        #         'p_loss': loss['losses']['p_loss'],
        #         'cos_loss': loss['losses']['cos_loss'],
        #         'sin_loss': loss['losses']['sin_loss'],
        #         'width_loss': loss['losses']['width_loss']
        #         }
        # result.log_dict(logs)
        result = {'loss':loss['loss']}
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

        # result = pl.EvalResult()
        # result.log('val_loss', loss['loss'], prog_bar=False)
        # result.log_dict({'train_loss': loss['loss'],
        #                  'p_loss': loss['losses']['p_loss'],
        #                  'cos_loss': loss['losses']['cos_loss'],
        #                  'sin_loss': loss['losses']['sin_loss'],
        #                  'width_loss': loss['losses']['width_loss']
        #                  })
        result = {'val_loss': loss['loss']}
        # result.correct = torch.tensor(float(s))
        # result.failed = torch.tensor(float(not s))
        result['log'] = {'correct':torch.tensor(float(s)), 'failed':torch.tensor(float(not s))}
        return result

    def validation_epoch_end(self, val_step_outputs):
        val_loss = torch.stack([x['val_loss'] for x in val_step_outputs]).mean()
        correct = torch.stack([x['log']['correct'] for x in val_step_outputs]).sum()
        failed = torch.stack([x['log']['failed'] for x in val_step_outputs]).sum()
        iou = torch.div(correct,correct+failed)
        result = {'val_loss':val_loss,'correct_sum':correct,'failed_sum':failed,'IoU':iou}
        print(f'IoU: {correct:.0f}/{correct+failed:.0f} = {iou:.2f}')
        return result

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
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1)
