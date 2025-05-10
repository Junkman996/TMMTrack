import torch
import torch.nn as nn
from .utils import diou_loss

class SmoothL1DIoULoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds, targets, anchors):
        # preds, targets: [B,N,4]  (dx,dy,dw,dh)
        # anchors: [B,N,4] absolute bbox coords
        # decode preds->bbox
        pred_boxes = self.decode(preds, anchors)
        target_boxes = self.decode(targets, anchors)
        l1 = self.smooth_l1(preds, targets).sum(-1)  # [B,N]
        diou = diou_loss(pred_boxes, target_boxes)  # [B,N]
        loss = self.alpha * l1 + self.beta * diou
        return loss.mean()

    @staticmethod
    def decode(deltas, anchors):
        # anchors: x1,y1,x2,y2 -> cx,cy,w,h
        x1,y1,x2,y2 = anchors.unbind(-1)
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w = x2-x1
        h = y2-y1
        dx,dy,dw,dh = deltas.unbind(-1)
        pred_cx = dx*w + cx
        pred_cy = dy*h + cy
        pred_w = torch.exp(dw) * w
        pred_h = torch.exp(dh) * h
        x1 = pred_cx - pred_w/2
        y1 = pred_cy - pred_h/2
        x2 = pred_cx + pred_w/2
        y2 = pred_cy + pred_h/2
        return torch.stack([x1,y1,x2,y2],dim=-1)