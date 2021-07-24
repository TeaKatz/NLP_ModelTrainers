from torch.nn import L1Loss, MSELoss, HuberLoss, CosineEmbeddingLoss

from ..BaseTrainerModule import BaseTrainerModule


class TeacherStudentTrainerModule(BaseTrainerModule):
    def __init__(self, teacher, student, loss=["mae"]):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss = loss if isinstance(loss, list) else [loss]

    def forward(self, teacher_inputs, student_inputs):
        teacher_outputs = self.teacher(teacher_inputs)
        student_outputs = self.student(student_inputs)
        return teacher_outputs, student_outputs

    def loss_func(self, teacher_outputs, student_outputs):
        loss = 0.
        if "mae" in self.loss:
            pass
        if "mse" in self.loss:
            pass
        if "huber" in self.loss:
            pass
        if "cosine" in self.loss:
            loss += CosineEmbeddingLoss(reduction="sum")(teacher_outputs, student_outputs, 1)
        return loss

    def cal_loss(self, outputs, targets):
        teacher_outputs, student_outputs = outputs

        teacher_outputs = teacher_outputs.float()
        student_outputs = student_outputs.float()
        return self.loss_func(teacher_outputs, student_outputs)

    def cal_metrics(self, outputs, targets):
        pass