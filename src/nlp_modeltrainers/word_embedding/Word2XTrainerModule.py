from nlp_losses import Losses
from ..BaseTrainerModule import BaseTrainerModule


class BinaryWord2XTrainerModule(BaseTrainerModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss_func(self, outputs, targets):
        return Losses("BinaryCrossEntropyLoss")(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, *)
        targets: (batch_size, *)
        """
        outputs = outputs.float()
        targets = targets.long()

        outputs = outputs.view(-1)
        targets = targets.view(-1)
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        pass


class MulticlassWord2XTrainerModule(BaseTrainerModule):
    def __init__(self, model, ignore_index=0):
        super().__init__()
        self.model = model
        self.ignore_index = ignore_index

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss_func(self, outputs, targets):
        return Losses("CategoricalCrossEntropyLoss", ignore_index=self.ignore_index)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, *, class_size)
        targets: (batch_size, *)
        """
        class_size = outputs.shape[-1]

        outputs = outputs.float()
        targets = targets.long()

        outputs = outputs.view(-1, class_size)
        targets = targets.view(-1)
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        pass


class MultilabelWord2XTrainerModule(BaseTrainerModule):
    def __init__(self, model, ignore_index=0):
        super().__init__()
        self.model = model
        self.ignore_index = ignore_index

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss_func(self, outputs, targets):
        return Losses("CategoricalCrossEntropyLoss", ignore_index=self.ignore_index)(outputs, targets)

    def cal_loss(self, outputs, targets):
        """
        outputs: (batch_size, *, class_size)
        targets: (batch_size, *, class_size)
        """
        class_size = outputs.shape[-1]

        outputs = outputs.float()
        targets = targets.long()

        outputs = outputs.view(-1, class_size)
        targets = targets.view(-1, class_size)
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        pass