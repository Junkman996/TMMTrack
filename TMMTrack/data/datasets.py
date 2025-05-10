import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from .utils import compute_iou

class MOTSequenceDataset(Dataset):
    def __init__(self, ann_file, img_dir, seq_len=10, transform=None):
        with open(ann_file) as f:
            self.annotations = yaml.safe_load(f)
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.sequences = list(self.annotations.keys())

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_id = self.sequences[idx]
        frames = self.annotations[seq_id]['frames']
        seq = []
        for fid in frames[:self.seq_len]:
            img_path = os.path.join(self.img_dir, seq_id, f"{fid}.jpg")
            img = read_image(img_path).float() / 255.0
            ann = frames[fid]['bboxes']  # [[x1,y1,x2,y2],...]
            if self.transform:
                img, ann = self.transform(img, ann)
            seq.append((img, torch.tensor(ann)))
        return seq