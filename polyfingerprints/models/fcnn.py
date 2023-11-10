from pytorch_lightning import LightningModule
import torch
import numpy as np
from torch.nn import MSELoss


class FCNNModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        reduction: float = 3,
        activation="ELU",
        lr: float = 1e-3,
    ):
        super().__init__()
        inputs = [input_dim]
        while int(np.ceil(inputs[-1] / reduction)) > output_dim:
            inputs.append(int(np.ceil(inputs[-1] / reduction)))

        activation = getattr(torch.nn, activation)

        layer = []

        for i in range(len(inputs) - 1):
            layer.append(torch.nn.Linear(inputs[i], inputs[i + 1]))
            layer.append(activation())
        layer.append(torch.nn.Linear(inputs[-1], output_dim))

        self.layers = torch.nn.Sequential(*layer)
        self.lr = lr
        self.loss_fn = MSELoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y)
        self.log("val_loss", val_loss)
        return {"val_loss": val_loss}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        test_loss = self.loss_fn(y_pred, y)
        self.log("test_loss", test_loss)
        return {"test_loss": test_loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
