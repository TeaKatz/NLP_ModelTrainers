from torch.nn import L1Loss

from ...Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class TeacherStudentTrainerModule(BaseTrainerModule):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def forward(self, teacher_inputs, student_inputs):
        teacher_outputs = self.teacher(teacher_inputs)
        student_outputs = self.student(student_inputs)
        return teacher_outputs, student_outputs

    @staticmethod
    def loss_func(teacher_outputs, student_outputs):
        """
        teacher_outputs: (batch_size, vector_size)
        student_outputs: (batch_size, vector_size)
        """
        return L1Loss(reduction="sum")(teacher_outputs, student_outputs)

    @staticmethod
    def metrics_func(teacher_outputs, student_outputs):
        return Metrics(["Cosine_Similarity"], names=["Cosine_Similarity"])(teacher_outputs, student_outputs)

    def cal_loss(self, outputs, targets):
        teacher_outputs, student_outputs = outputs

        teacher_outputs = teacher_outputs.float()
        student_outputs = student_outputs.float()
        return self.loss_func(teacher_outputs, student_outputs)

    def cal_metrics(self, outputs, targets):
        teacher_outputs, student_outputs = outputs

        teacher_outputs = teacher_outputs.cpu().detach().numpy()
        student_outputs = student_outputs.cpu().detach().numpy()

        teacher_outputs = teacher_outputs.astype(float)
        student_outputs = student_outputs.astype(float)
        return self.metrics_func(teacher_outputs, student_outputs)
