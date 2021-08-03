from nlp_losses import Losses
from nlp_metrics import Metrics
from ..BaseTrainerModule import BaseTrainerModule


class TeacherStudentTrainerModule(BaseTrainerModule):
    def __init__(self, teacher, student):
        super().__init__()
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
    def __init__(self, student):
        super().__init__()
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
