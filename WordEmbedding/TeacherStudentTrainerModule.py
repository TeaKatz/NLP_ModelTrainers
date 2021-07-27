from ...Losses import Losses
from ...Metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class TeacherStudentTrainerModule(BaseTrainerModule):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
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
        return Losses(["MAE"])(teacher_outputs, student_outputs)

    @staticmethod
    def metrics_func(student_outputs, teacher_outputs):
        """
        student_outputs: (batch_size, vector_size)
        teacher_outputs: (batch_size, vector_size)
        """
        return Metrics(["Cosine_Similarity"])(teacher_outputs, student_outputs)

    def cal_loss(self, outputs, targets):
        student_outputs, teacher_outputs = outputs

        student_outputs = student_outputs.float()
        teacher_outputs = teacher_outputs.float()
        return self.loss_func(student_outputs, teacher_outputs)

    def cal_metrics(self, outputs, targets):
        student_outputs, teacher_outputs = outputs

        student_outputs = student_outputs.cpu().detach().numpy()
        teacher_outputs = teacher_outputs.cpu().detach().numpy()

        student_outputs = student_outputs.astype(float)
        teacher_outputs = teacher_outputs.astype(float)
        return self.metrics_func(student_outputs, teacher_outputs)
