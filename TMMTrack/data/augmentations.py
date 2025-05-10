import random
import torch

def random_jitter(image, bboxes, scale=0.05):
    _, h, w = image.shape
    new_bboxes = []
    for box in bboxes:
        x1, y1, x2, y2 = box
        dx = (random.random() - 0.5) * scale * w
        dy = (random.random() - 0.5) * scale * h
        new_box = [x1+dx, y1+dy, x2+dx, y2+dy]
        new_bboxes.append(new_box)
    return image, torch.tensor(new_bboxes)

def erase_random_segment(sequence, erase_ratio=0.2):
    L = len(sequence)
    seg_len = max(1, int(L * erase_ratio))
    start = random.randint(0, L-seg_len)
    for i in range(start, start+seg_len):
        img, _ = sequence[i]
        sequence[i] = (torch.zeros_like(img), sequence[i][1])
    return sequence