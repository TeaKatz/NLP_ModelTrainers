from nlp_losses import Losses
from ..BaseTrainerModule import BaseTrainerModule


class Word2XTrainerModule(BaseTrainerModule):
    output_type_options = ["binary", "multiclass", "multilabel"]

    def __init__(self, model, output_type="multiclass", ignore_index=0):
        assert output_type in self.output_type_options, "output_type is incorrect"

        super().__init__()
        self.model = model
        self.output_type = output_type
        self.ignore_index = ignore_index

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def loss_func(self, outputs, targets):
        if self.output_type == "multiclass":
            return Losses("CategoricalCrossEntropyLoss", ignore_index=self.ignore_index)(outputs, targets)
        else:
            return Losses("BinaryCrossEntropyLoss", ignore_index=self.ignore_index)(outputs, targets)

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