import torch
from abc import abstractmethod
from pytorch_lightning import LightningModule


class BaseTrainerModule(LightningModule):
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def cal_loss(self, outputs, targets):
        pass

    @abstractmethod
    def cal_metrics(self, outputs, targets):
        pass

    def training_step(self, batch, *args, **kwargs):
        if isinstance(batch["input"], dict):
            outputs = self(**batch["input"])
        else:
            outputs = self(batch["input"])

        # Compute loss and metrics
        if isinstance(batch["target"], dict):
            loss = self.cal_loss(outputs, **batch["target"])
            metrics = self.cal_metrics(outputs, **batch["target"])
        else:
            loss = self.cal_loss(outputs, batch["target"])
            metrics = self.cal_metrics(outputs, batch["target"])

        # Record loss and metrics
        self.log("train_loss", loss, prog_bar=True, logger=True)
        if metrics is not None:
            for name, value in metrics.items():
                self.log(f"train_{name.lower()}", value, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, *args, **kwargs):
        if isinstance(batch["input"], dict):
            outputs = self(**batch["input"])
        else:
            outputs = self(batch["input"])

        # Compute loss and metrics
        if isinstance(batch["target"], dict):
            loss = self.cal_loss(outputs, **batch["target"])
            metrics = self.cal_metrics(outputs, **batch["target"])
        else:
            loss = self.cal_loss(outputs, batch["target"])
            metrics = self.cal_metrics(outputs, batch["target"])

        # Record loss and metrics
        self.log("val_loss", loss, prog_bar=True, logger=True)
        if metrics is not None:
            for name, value in metrics.items():
                self.log(f"val_{name.lower()}", value, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, *args, **kwargs):
        if isinstance(batch["input"], dict):
            outputs = self(**batch["input"])
        else:
            outputs = self(batch["input"])

        # Compute loss and metrics
        if isinstance(batch["target"], dict):
            loss = self.cal_loss(outputs, **batch["target"])
            metrics = self.cal_metrics(outputs, **batch["target"])
        else:
            loss = self.cal_loss(outputs, batch["target"])
            metrics = self.cal_metrics(outputs, batch["target"])

        # Record loss and metrics
        self.log("test_loss", loss, prog_bar=True, logger=False)
        if metrics is not None:
            for name, value in metrics.items():
                self.log(f"test_{name.lower()}", value, prog_bar=True, logger=False)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num")
        items.pop("loss")
        return items
