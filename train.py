import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from sklearn.metrics import accuracy_score

from model import ConvNet, ConfidNet
from utils import ConfidenceHistograms


class Trainer:
    def __init__(
        self,
        seed=23,
        log_dir="~/experiments/ConfidNet",
        train_val_split=.8,
        device="cuda",
        model_checkpoint=10,
        convnet_kwargs={},
        confidnet_kwargs={},
        optimizer_kwargs={},
        loader_kwargs={},
    ):
        self.seed = seed
        self.train_val_split = train_val_split
        self.device = device
        self.model_checkpoint = model_checkpoint
        self.convnet_kwargs = convnet_kwargs
        self.confidnet_kwargs = confidnet_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.loader_kwargs = loader_kwargs

        self.log_dir = Path(log_dir).expanduser()
        self.model_filename = self.log_dir / "model"
        self.model_filename.mkdir(parents=True, exist_ok=True)
        self.model_filename = self.model_filename / "{model}_{epoch}.pth"

    def get_datasets(self):
        torch.manual_seed(self.seed)
        train_valid_dataset = torchvision.datasets.MNIST(
            root=Path("~/datasets/MNIST").expanduser(),
            train=True,
            transform=T.ToTensor(),
            download=True)
        nb_train = int(self.train_val_split * len(train_valid_dataset))
        nb_valid = len(train_valid_dataset) - nb_train
        train_dataset, valid_dataset = torch.utils.data.dataset.random_split(
            train_valid_dataset, [nb_train, nb_valid])
        return train_dataset, valid_dataset

    def train_convnet(self, epoch=100, epoch_to_restore=0):
        train, val = self.get_datasets()
        train_loader = DataLoader(train, **self.loader_kwargs)
        val_loader = DataLoader(val, **self.loader_kwargs)

        net = ConvNet(**self.convnet_kwargs).to(self.device)
        optimizer = Adam(net.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        writer = SummaryWriter(self.log_dir)

        for e in range(epoch_to_restore + 1, epoch + epoch_to_restore + 1):
            net.train()
            history_train = {"loss": [], "metric": []}
            for idx_batch, (imgs, label) in enumerate(train_loader):
                imgs, label = imgs.to(self.device), label.to(self.device)

                net.zero_grad()
                pred, _ = net(imgs)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                history_train["loss"].append(loss.detach().item())

                pred = F.softmax(pred.detach(), dim=1).argmax(1)
                score = accuracy_score(label.cpu(), pred.cpu())
                history_train["metric"].append(score)

                step = min((idx_batch+1) * self.loader_kwargs["batch_size"], len(train))
                sys.stdout.write(
                    "Training : "
                    f"Epoch {e}/{epoch + epoch_to_restore}; "
                    f"Step {step}/{len(train)}; "
                    f"Loss {loss.detach().item()}; "
                    f"Score {score}\r"
                )
                sys.stdout.flush()
            print()
            print()

            net.eval()
            history_val = {"loss": [], "metric": []}
            for idx_batch, (imgs, label) in enumerate(val_loader):
                imgs, label = imgs.to(self.device), label.to(self.device)

                with torch.no_grad():
                    pred, _ = net(imgs)
                loss = criterion(pred, label).detach().item()
                history_val["loss"].append(loss)

                pred = F.softmax(pred.detach(), dim=1).argmax(1)
                score = accuracy_score(label.cpu(), pred.cpu())
                history_val["metric"].append(score)

                step = min((idx_batch+1) * self.loader_kwargs["batch_size"], len(train))
                sys.stdout.write(
                    "Validation : "
                    f"Epoch {e}/{epoch + epoch_to_restore}; "
                    f"Step {step}/{len(train)}; "
                    f"Loss {loss}; "
                    f"Score {score}\r"
                )
                sys.stdout.flush()

            writer.add_scalars(
                'ConvNet/Loss',
                {"train": np.mean(history_train["loss"]), "val": np.mean(history_val["loss"])},
                e)
            writer.add_scalars(
                'ConvNet/Accuracy',
                {"train": np.mean(history_train["metric"]), "val": np.mean(history_val["metric"])},
                e)

            print(
                f"\n\n[*] Finished epoch {e};\n\n"
                "Train :\n"
                f"\tLoss : {np.mean(history_train['loss'])}\n"
                f"\tAccuracy : {np.mean(history_train['metric'])}\n"
                "Test :\n"
                f"\tLoss : {np.mean(history_val['loss'])}\n"
                f"\tAccuracy : {np.mean(history_val['metric'])}\n\n\n"
            )

            if e % self.model_checkpoint == 0:
                filename = str(self.model_filename).format(model="ConvNet", epoch=e)
                torch.save(net.state_dict(), filename)

        return filename

    def train_confidnet(self, convnet_path, epoch=100, epoch_to_restore=0):
        train, val = self.get_datasets()
        train_loader = DataLoader(train, **self.loader_kwargs)
        val_loader = DataLoader(val, **self.loader_kwargs)

        conv_net = ConvNet(**self.convnet_kwargs).to(self.device)
        conv_net.load_state_dict(torch.load(convnet_path))
        conv_net.eval()
        confid_net = ConfidNet(**self.confidnet_kwargs).to(self.device)
        optimizer = Adam(confid_net.parameters(), **self.optimizer_kwargs)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
        criterion = torch.nn.MSELoss()
        writer = SummaryWriter(self.log_dir)

        for e in range(epoch_to_restore + 1, epoch + epoch_to_restore + 1):
            confid_net.train()
            history_train = {"loss": [], "metric": []}
            train_histograms = ConfidenceHistograms()
            for idx_batch, (imgs, label) in enumerate(train_loader):
                imgs, label = imgs.to(self.device), label.to(self.device)
                with torch.no_grad():
                    pred, encoded = conv_net(imgs)

                confid_net.zero_grad()
                confidence = confid_net(encoded)
                loss = criterion(confidence, F.softmax(pred, dim=1).gather(1, label.unsqueeze(1)))
                loss.backward()
                optimizer.step()
                train_histograms.step(label, pred, confidence.detach())
                history_train["loss"].append(loss.detach().item())

                step = min((idx_batch+1) * self.loader_kwargs["batch_size"], len(train))
                sys.stdout.write(
                    "Training : "
                    f"Epoch {e}/{epoch + epoch_to_restore}; "
                    f"Step {step}/{len(train)}; "
                    f"Loss {loss.detach().item()};\r"
                )
                sys.stdout.flush()
            print()
            print()

            confid_net.eval()
            history_val = {"loss": [], "metric": []}
            val_histograms = ConfidenceHistograms()
            for idx_batch, (imgs, label) in enumerate(val_loader):
                imgs, label = imgs.to(self.device), label.to(self.device)

                with torch.no_grad():
                    pred, encoded = conv_net(imgs)
                    confidence = confid_net(encoded)
                val_histograms.step(label, pred, confidence)
                loss = criterion(confidence, F.softmax(pred, dim=1).gather(1, label.unsqueeze(1)))
                history_val["loss"].append(loss.detach().item())

                step = min((idx_batch+1) * self.loader_kwargs["batch_size"], len(train))
                sys.stdout.write(
                    "Validation : "
                    f"Epoch {e}/{epoch + epoch_to_restore}; "
                    f"Step {step}/{len(train)}; "
                    f"Loss {loss.detach().item()};\r"
                )
                sys.stdout.flush()

            # scheduler.step(np.mean(history_val["loss"]))
            train_mcp_hist, train_tcp_hist = train_histograms.get_histograms()
            writer.add_figure("ConfidNet/train/MCP", train_mcp_hist, e)
            writer.add_figure("ConfidNet/train/TCP", train_tcp_hist, e)
            val_mcp_hist, val_tcp_hist = val_histograms.get_histograms()
            writer.add_figure("ConfidNet/val/MCP", val_mcp_hist, e)
            writer.add_figure("ConfidNet/val/TCP", val_tcp_hist, e)
            writer.add_scalars(
                'ConfidNet/Loss',
                {"train": np.mean(history_train["loss"]), "val": np.mean(history_val["loss"])},
                e)

            print(
                f"\n\n[*] Finished epoch {e};\n\n"
                "Train :\n"
                f"\tLoss : {np.mean(history_train['loss'])}\n"
                "Test :\n"
                f"\tLoss : {np.mean(history_val['loss'])}\n\n\n"
            )

            if e % self.model_checkpoint == 0:
                filename = str(self.model_filename).format(model="ConfidNet", epoch=e)
                torch.save(confid_net.state_dict(), filename)

        return confid_net
