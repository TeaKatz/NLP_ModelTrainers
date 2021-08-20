import numpy as np

from nlp_losses import Losses
from nlp_metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class TeacherStudentTrainerModule(BaseTrainerModule):
    def __init__(self, teacher, student, learning_rate=1e-3):
        super().__init__(learning_rate)
        self.teacher = teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.student = student

    def forward(self, student_inputs, teacher_inputs):
        student_outputs = self.student(student_inputs)
        teacher_outputs = self.teacher(teacher_inputs)
        return student_outputs, teacher_outputs

    @staticmethod
    def loss_func(student_outputs, teacher_outputs):
        """
        student_outputs: (batch_size, vector_size)
        teacher_outputs: (batch_size, vector_size)
        """
        return Losses(["MAELoss"])(student_outputs, teacher_outputs)

    @staticmethod
    def metrics_func(student_outputs, teacher_outputs):
        """
        student_outputs: (batch_size, vector_size)
        teacher_outputs: (batch_size, vector_size)
        """
        return Metrics(["Cosine_Similarity"])(student_outputs, teacher_outputs)

    def cal_loss(self, outputs, targets=None):
        student_outputs, teacher_outputs = outputs

        student_outputs = student_outputs.float()
        teacher_outputs = teacher_outputs.float()
        return self.loss_func(student_outputs, teacher_outputs)

    def cal_metrics(self, outputs, targets=None):
        student_outputs, teacher_outputs = outputs

        student_outputs = student_outputs.cpu().detach().numpy()
        teacher_outputs = teacher_outputs.cpu().detach().numpy()

        student_outputs = student_outputs.astype(float)
        teacher_outputs = teacher_outputs.astype(float)
        return self.metrics_func(student_outputs, teacher_outputs)


class StudentTrainerModule(BaseTrainerModule):
    def __init__(self, student, learning_rate=1e-3):
        super().__init__(learning_rate)
        self.student = student

    def forward(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    @staticmethod
    def loss_func(outputs, targets):
        """
        outputs: (batch_size, vector_size)
        targets: (batch_size, vector_size)
        """
        return Losses(["MAELoss"])(outputs, targets)

    @staticmethod
    def metrics_func(outputs, targets):
        """
        outputs: (batch_size, vector_size)
        targets: (batch_size, vector_size)
        """
        return Metrics(["Cosine_Similarity"])(outputs, targets)

    def cal_loss(self, outputs, targets):
        outputs = outputs.float()
        targets = targets.float()
        return self.loss_func(outputs, targets)

    def cal_metrics(self, outputs, targets):
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        outputs = outputs.astype(float)
        targets = targets.astype(float)
        return self.metrics_func(outputs, targets)


class StudentWithNegativeSamplesTrainerModule(BaseTrainerModule):
    def __init__(self, student, learning_rate=1e-3):
        super().__init__(learning_rate)
        self.student = student

    def forward(self, *args, **kwargs):
        return self.student(*args, **kwargs)

    @staticmethod
    def loss_func(outputs, positives, negatives):
        """
        outputs: (batch_size, vector_size)
        positives: (batch_size, vector_size)
        negatives: (batch_size, negative_size, vector_size)
        """
        return Losses(["FastTextLoss"])(outputs, positives, negatives)

    @staticmethod
    def metrics_func(outputs, positives, negatives):
        """
        outputs: (batch_size, vector_size)
        positives: (batch_size, vector_size)
        negatives: (batch_size, negative_size, vector_size)
        """
        _, negative_size, _ = negatives.shape

        metrics = {}
        # Positive
        pos_metrics = Metrics(["Cosine_Similarity"])(outputs, positives)["Cosine_Similarity"]
        metrics["Positive_Similarity"] = pos_metrics

        # Negative
        neg_metrics = []
        for i in range(negative_size):
            neg_metrics.append(Metrics(["Cosine_Similarity"])(outputs, negatives[:, i])["Cosine_Similarity"])
        neg_metrics = np.mean(neg_metrics)
        metrics["Negative_Similarity"] = neg_metrics
        return metrics

    def cal_loss(self, outputs, targets):
        positives, negatives = targets

        outputs = outputs.float()
        positives = positives.float()
        negatives = negatives.float()
        return self.loss_func(outputs, positives, negatives)

    def cal_metrics(self, outputs, targets):
        positives, negatives = targets

        outputs = outputs.cpu().detach().numpy()
        positives = positives.cpu().detach().numpy()
        negatives = negatives.cpu().detach().numpy()

        outputs = outputs.astype(float)
        positives = positives.astype(float)
        negatives = negatives.astype(float)
        return self.metrics_func(outputs, positives, negatives)
