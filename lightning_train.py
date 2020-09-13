import pytorch_lightning as pl
from argparse import ArgumentParser
import os

from inference.models.grconvnet_lightning import GraspModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


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

class CheckpointEveryNSteps(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        save_step_frequency,
        prefix="N-Step-Checkpoint",
        use_modelcheckpoint_filename=False,
    ):
        """
        Args:
            save_step_frequency: how often to save in steps
            prefix: add a prefix to the name, only used if
                use_modelcheckpoint_filename=False
            use_modelcheckpoint_filename: just use the ModelCheckpoint callback's
                default filename, don't use ours.
        """
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_batch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train batch """
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch}_{global_step}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)

checkpoint_callback = ModelCheckpoint(
    filepath='trained-models/',
    save_last=True,
    # save_top_k=1,
    verbose=True,
    monitor='val_loss',
    mode='min',
    prefix='grconvnet_'
)

wandb_logger = WandbLogger(name='GRConvnet-ch32-jacquard', project='IROS-grasping')
nstep_ckpt_cb = CheckpointEveryNSteps(1000)

trainer = pl.Trainer.from_argparse_args(args,
                                        checkpoint_callback=checkpoint_callback,
                                        # logger=wandb_logger,
                                        callbacks=[nstep_ckpt_cb]
                                        )
model = GraspModule(args)
trainer.fit(model)