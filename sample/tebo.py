from torch.utils.tensorboard import SummaryWriter
import deepxde as dde

writer = SummaryWriter('runs/experiment')

class TensorBoardCallback(dde.callbacks.Callback):
    def __init__(self, writer):
        super().__init__()
        self.writer = writer
        
    def on_epoch_end(self):
        # 记录训练损失
        self.writer.add_scalar('Loss/train', self.model.train_state.loss_train, self.model.train_state.step)
        
        # 记录测试指标（如果有的话）
        if self.model.train_state.metrics_test:
            self.writer.add_scalar('Metrics/l2_error', self.model.train_state.metrics_test[0], self.model.train_state.step)
        
        self.writer.flush()