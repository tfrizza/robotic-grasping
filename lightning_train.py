import pytorch_lightning as pl
from argparse import ArgumentParser
import os

# from inference.lightning_grasp import GraspModule
from inference.models.grconvnet_lightning import GenerativeResnet
from utils.data.lightning_data import GraspDataModule
from pytorch_lightning.callbacks import ModelCheckpoint

def parse_args():
    parser = ArgumentParser(description='Train network')

    # Network
    parser.add_argument('--network', type=str, default='grconvnet3',
                        help='Network name in inference/models')
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

    # Datasets
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
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Training
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--batches-per-epoch', type=int, default=1000,
                        help='Batches per Epoch')
    parser.add_argument('--optim', type=str, default='adam',
                        help='Optmizer for the training. (adam or SGD)')
    parser.add_argument('--tpu', action='store_true', default=False,
                        help='Whether to run on tpu or not')

    # Logging etc.
    parser.add_argument('--description', type=str, default='',
                        help='Training description')
    parser.add_argument('--logdir', type=str, default='logs/',
                        help='Log directory')
    parser.add_argument('--vis', action='store_true',
                        help='Visualise the training process')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()
    return args

args = parse_args()

# Load the network
input_channels = 1 * args.use_depth + 3 * args.use_rgb
grasp_model = GenerativeResnet(args,
    input_channels=input_channels,
    dropout=args.use_dropout,
    prob=args.dropout_prob,
    channel_size=args.channel_size
)

checkpoint_cb = ModelCheckpoint(
    filepath=os.getcwd(),
    save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix=''
)

# trainer = pl.Trainer(tpu_cores=8, precision=32, auto_lr_find=False)
tpu_cores = 8 if args.tpu else None
trainer = pl.Trainer(tpu_cores=tpu_cores, checkpoint_callback=checkpoint_cb)
trainer.fit(grasp_model)