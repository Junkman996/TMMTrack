import torch

def diou_loss(pred, target, eps=1e-7):
    # pred, target: [B,N,4]
    x1_p,y1_p,x2_p,y2_p = pred.unbind(-1)
    x1_t,y1_t,x2_t,y2_t = target.unbind(-1)
    # intersection
    xi1, yi1 = torch.max(x1_p, x1_t), torch.max(y1_p, y1_t)
    xi2, yi2 = torch.min(x2_p, x2_t), torch.min(y2_p, y2_t)
    inter_area = (xi2-xi1).clamp(0) * (yi2-yi1).clamp(0)
    # union
    area_p = (x2_p-x1_p) * (y2_p-y1_p)
    area_t = (x2_t-x1_t) * (y2_t-y1_t)
    union = area_p + area_t - inter_area + eps
    iou = inter_area / union
    # center distance
    cx_p = (x1_p+x2_p)/2; cy_p = (y1_p+y2_p)/2
    cx_t = (x1_t+x2_t)/2; cy_t = (y1_t+y2_t)/2
    dist2 = (cx_p-cx_t)**2 + (cy_p-cy_t)**2
    # enclose box
    xc1, yc1 = torch.min(x1_p, x1_t), torch.min(y1_p, y1_t)
    xc2, yc2 = torch.max(x2_p, x2_t), torch.max(y2_p, y2_t)
    cw2 = (xc2-xc1)**2 + (yc2-yc1)**2 + eps
    diou = 1 - iou + dist2 / cw2
    return diou