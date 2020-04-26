import os
import argparse

import torch
import joblib

from train import Trainer


parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',)
parser.add_argument('-s', '--seed', type=int, default=23,
                    help='[int] Manual seed',)
parser.add_argument('-e', '--epochs', type=int, default=100,
                    help='[int] Number of epochs',)
parser.add_argument('-bs', '--batch-size', type=int, default=128,
                    help='[int] Batch size',)
parser.add_argument('-nw', '--num-workers', type=int, default=8,
                    help='[int] Number of workers',)
parser.add_argument('-cp', '--checkpoints', type=int, default=10,
                    help='[int] Model checkpoints',)
parser.add_argument('-ts', '--train-split', type=float, default=.8,
                    help='[float] Train split',)
parser.add_argument('-d', '--device', type=str, default="cuda",
                    choices=["cuda", "cpu"],
                    help='[str] Device used',)
parser.add_argument('-log', '--log-dir', type=str, default=None,
                    help='[str] Log directory',)
args = parser.parse_args()

loader_kwargs = {
    "batch_size": args.batch_size,
    "shuffle": True,
    "num_workers": args.num_workers,
    "pin_memory": True,
    "drop_last": True,
}

confidnet_kwargs = {
    "small": True,
}


parameters = dict(
    seed=args.seed,
    log_dir=args.log_dir,
    train_val_split=args.train_split,
    device=args.device,
    model_checkpoint=args.checkpoints,
    convnet_kwargs={},
    confidnet_kwargs={},
    loader_kwargs=loader_kwargs,
)

config_dir = args.log_dir if args.log_dir else "/tmp"
with open(os.path.join(config_dir, "config.jbl"), 'wb') as outfile:
    joblib.dump(parameters, outfile)

torch.manual_seed(args.seed)
trainer = Trainer(**parameters)
convnet_weight_path = trainer.train_convnet(
    epoch=args.epochs,
    # epoch_to_restore=args.epoch_to_restore
)
confidnet_weight_path = trainer.train_confidnet(
    convnet_weight_path,
    epoch=args.epochs,
)
