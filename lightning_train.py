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

# checkpoint_callback = ModelCheckpoint(
#     filepath=os.getcwd(),
#     save_top_k=True,
#     verbose=True,
#     save_weights_only=False,
#     monitor='val_loss',
#     mode='min',
#     prefix=''
# )

checkpoint_callback = ModelCheckpoint(
    filepath='/content/robotic-grasping/trained-models/',
    save_last=True,
    # save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='grasp'
)

trainer = pl.Trainer.from_argparse_args(args,
                                        checkpoint_callback=checkpoint_callback)
model = GraspModule(args)
trainer.fit(model)