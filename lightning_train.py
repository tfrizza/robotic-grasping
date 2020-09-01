import pytorch_lightning as pl
from argparse import ArgumentParser
import os

from inference.models.grconvnet_lightning import GraspModule
from pytorch_lightning.callbacks import ModelCheckpoint

parser = ArgumentParser()

# add model specific args
parser = GraspModule.add_model_specific_args(parser)

# add all the available trainer options to argparse
# e.g. --tpu_cores 8
parser = pl.Trainer.add_argparse_args(parser)

args = parser.parse_args()

trainer = pl.Trainer.from_argparse_args(args)
model = GraspModule(args)
trainer.fit(model)