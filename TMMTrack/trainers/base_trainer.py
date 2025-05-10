import os
import torch
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    def __init__(self, model, optimizer, scheduler, device, work_dir):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.work_dir = work_dir
        self.writer = SummaryWriter(log_dir=os.path.join(work_dir, 'logs'))

    def save_checkpoint(self, epoch, best=False):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
        }
        fname = f"checkpoint_{epoch}.pth" if not best else 'best.pth'
        torch.save(state, os.path.join(self.work_dir, fname))

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for i, data in enumerate(dataloader):
            loss = self.step(data)
            total_loss += loss.item()
            if i % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        self.scheduler.step()
        return total_loss / len(dataloader)

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in dataloader:
                loss = self.step(data, train=False)
                total_loss += loss.item()
        return total_loss / len(dataloader)

    def step(self, data, train=True):
        raise NotImplementedError