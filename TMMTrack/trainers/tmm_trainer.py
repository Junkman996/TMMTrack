import torch
from .base_trainer import BaseTrainer
from losses.smooth_l1_iou_loss import SmoothL1DIoULoss
from trainers.metrics import MetricLogger

class TMMTrainer(BaseTrainer):
    def __init__(self, model, optimizer, scheduler, device, work_dir):
        super().__init__(model, optimizer, scheduler, device, work_dir)
        self.criterion = SmoothL1DIoULoss()
        self.metric_logger = MetricLogger()
        self.global_step = 0

    def step(self, data, train=True):
        seq = [ [frame.to(self.device) for frame in layer] for layer in data ]
        # unpack sequence into tensors
        imgs = torch.stack([f[0] for f in seq], dim=1)            # [B,T,C,H,W]
        targets = torch.stack([f[1] for f in seq], dim=1)         # [B,T,N,4]
        # detection
        feats = self.model.tim(targets)  # dummy input
        preds = self.model.motion_predictor(feats)
        anchors = targets[:,-1]  # last frame as anchors
        loss = self.criterion(preds, torch.zeros_like(preds), anchors)
        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.global_step += 1
        # metrics
        # compute dummy distances for metric logger
        # self.metric_logger.update(gt_ids, pred_ids, distances)
        return loss