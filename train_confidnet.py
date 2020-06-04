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
parser.add_argument('-inc', '--in_channels', type=int, default=1,
                    help='[int] Input channels',)
parser.add_argument('-nc', '--num_classes', type=int, default=10,
                    help='[int] Input channels',)
parser.add_argument('-ts', '--train-split', type=float, default=.8,
                    help='[float] Train split',)
parser.add_argument('-lr', '--learning-rate', type=float, default=.001,
                    help='[float] Learning rate',)
parser.add_argument('-d', '--device', type=str, default="cuda",
                    choices=["cuda", "cpu"],
                    help='[str] Device used',)
parser.add_argument('-data', type=str, default='mnist',
                    help='[str] Dataset to use')
parser.add_argument('-convnet', type=str, required=True,
                    help="[str] Path to ConvNet's weights")
parser.add_argument('-log', '--log-dir', type=str, default=None,
                    help='[str] Log directory',)
parser.add_argument('-small', default=False, action="store_true",
                    help="Pass this argument to have a smaller ConfidNet")
args = parser.parse_args()

loader_kwargs = {
    "batch_size": args.batch_size,
    "shuffle": True,
    "num_workers": args.num_workers,
    "pin_memory": True,
    "drop_last": True,
}

confidnet_kwargs = {
    "small": args.small,
}

optimizer_kwargs = {
    "lr": args.learning_rate,
    "betas": (0.9, 0.999),
    "eps": 1e-08,
    "weight_decay": 0,
    "amsgrad": False,
}


parameters = dict(
    seed=args.seed,
    log_dir=args.log_dir,
    train_val_split=args.train_split,
    device=args.device,
    model_checkpoint=args.checkpoints,
    convnet_kwargs={},
    confidnet_kwargs=confidnet_kwargs,
    optimizer_kwargs=optimizer_kwargs,
    loader_kwargs=loader_kwargs,
)

config_dir = args.log_dir if args.log_dir else "/tmp"
with open(os.path.join(config_dir, "config.jbl"), 'wb') as outfile:
    joblib.dump(parameters, outfile)

torch.manual_seed(args.seed)
trainer = Trainer(**parameters)
confidnet_weight_path = trainer.train_confidnet(
    args.convnet,
    epoch=args.epochs,
)
