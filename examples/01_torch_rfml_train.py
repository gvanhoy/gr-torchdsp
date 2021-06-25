# RFML training script, patterned after https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb
# Dataset downloaded from here: http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2

from typing import Tuple, Optional
from torch import nn, utils
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch
import time
import pickle


class RMLDataset(torch.utils.data.Dataset):
    """
    Description of RMLDataset:
    A Python generator wrapped around the RML dataset.

    Attributes:
        data_file_path (type):
        index (type):

    Args:
        index_file_path (str): relative file path to .pkl containing the index
        data_file_path (str): relative file path to the .f32 containing all of the raw IQ data

    """

    def __init__(self, index_file_path: str, data_file_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.data_file_path = data_file_path
        self.index = pickle.load(open(index_file_path, 'rb'))

    def __getitem__(self, index: int) -> tuple:
        index_row = self.index[index]
        data = np.fromfile(self.data_file_path, dtype=np.float32,
                           count=256, offset=index_row[4]).reshape(2, 128)
        label = index_row[2]

        return data, label

    def __len__(self) -> int:
        return len(self.index)


class RMLDataModule(pl.LightningDataModule):
    """
    Description of RFMLDataModule:
    Data module containing train/test versions of the RML dataset

    Attributes:
        dataset (type):

    Inheritance:
        pl.LightningDataModule:

    """

    def __init__(self) -> None:
        super().__init__()
        self.dataset = RMLDataset("rml_index.pkl", "rml_data.f32")

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.test_dataset = random_split(
            self.dataset,
            lengths=[110000, 110000],
            generator=torch.Generator().manual_seed(2019)
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=1024, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1024, num_workers=8)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=1024, num_workers=8)


class VTCNNModule(pl.LightningModule):
    """
    Description of VTCNNModule:
    A Lightning Module containing a network with the VT-CNN2 architecture.

    Attributes:
        model (type):

    Inheritance:
        pl.LightningModule:

    """

    def __init__(self) -> None:
        super().__init__()
        conv1 = nn.Conv2d(1, 256, (2, 3), padding=(0, 2))
        torch.nn.init.xavier_uniform_(
            conv1.weight, gain=nn.init.calculate_gain('relu')
        )
        conv2 = nn.Conv2d(256, 80, (1, 3), padding=(0, 2))
        torch.nn.init.xavier_uniform_(
            conv2.weight, gain=nn.init.calculate_gain('relu')
        )

        self.model = torch.nn.Sequential(
            conv1,
            nn.ReLU(),
            nn.Dropout(.5),
            conv2,
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Flatten(),
            nn.Linear(10560, 256),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(256, 11)
        )

    def on_epoch_end(self) -> None:
        # This was added because training at full speed actually caused the machine to overheat and shut down.
        time.sleep(10)
        return super().on_epoch_end()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # this was not done in the original, but this does not train well without it and this should be done with a real radio.
        x -= x.min()
        x /= x.max() - x.min()
        x = 2*x - 1
        x = torch.unsqueeze(x, 1)
        # x = torch.reshape(x, (-1, 1, 2, 128))
        return self.model(x)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, eps=1e-7)

        return optimizer

        # # The scheduler was also not done in the original, but is helpful in stabilizing the learning over a long time
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer, gamma=.995
        # )
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': scheduler
        #     }
        # }

    def training_step(self, train_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        data, ground_truth = train_batch
        output = self.forward(data)

        loss = F.cross_entropy(output, ground_truth)

        pred = output.argmax(dim=1)  # get the index of the max log-probability
        acc = pred.eq(ground_truth).sum().item() * 100 / ground_truth.shape[0]

        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, val_batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        data, ground_truth = val_batch
        output = self.forward(data)

        loss = F.cross_entropy(output, ground_truth)

        pred = output.argmax(dim=1)  # get the index of the max log-probability
        acc = pred.eq(ground_truth).sum().item() * 100 / ground_truth.shape[0]

        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        return loss


if __name__ == "__main__":
    trainer = pl.Trainer(gpus=1)
    trainer.fit(VTCNNModule(), RMLDataModule())
