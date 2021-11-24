import torch
from abc import abstractmethod
from pytorch_lightning import LightningModule


class BaseTrainerModule(LightningModule):
    def __init__(self, 
                learning_rate=1e-3, 
                optimizer=None, 
                optimizer_params=None, 
                lr_scheduler=None, 
                lr_scheduler_params=None, 
                lr_scheduler_config=None):

        super().__init__()
        self.optimizer = torch.optim.Adam if optimizer is None else optimizer
        self.optimizer_params = {"lr": learning_rate} if optimizer_params is None else optimizer_params
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_params = {} if lr_scheduler_params is None else lr_scheduler_params
        self.lr_scheduler_config = {} if lr_scheduler_config is None else lr_scheduler_config
        self.save_hyperparameters()

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
        config = {"optimizer": self.optimizer(self.parameters(), **self.optimizer_params)}
        if self.lr_scheduler is not None:
            config["lr_scheduler"] = {
                "scheduler": self.lr_scheduler(config["optimizer"], **self.lr_scheduler_params),
                **self.lr_scheduler_config
            }
        return config

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num")
        items.pop("loss")
        return items
