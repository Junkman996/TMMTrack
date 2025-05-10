import torch
from yolox.exp import get_exp
from yolox.utils import postprocess

class YOLOXDetector:
    def __init__(self, exp_file, ckpt_file, device='cuda'):
        exp = get_exp(exp_file)
        self.model = exp.get_model().to(device)
        self.device = device
        ckpt = torch.load(ckpt_file, map_location=device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

    @torch.no_grad()
    def detect(self, img):
        img = img.to(self.device)
        outputs = self.model(img.unsqueeze(0))
        return postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)[0]
