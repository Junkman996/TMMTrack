import torch
from models.yolo_detector import YOLOXDetector
from models.tmm_module import TMMModule
from models.motion_predictor import MotionPredictor
from scipy.optimize import linear_sum_assignment

class Predictor:
    def __init__(self, det_cfg, det_ckpt, model_ckpt, device='cuda'):
        self.detector = YOLOXDetector(det_cfg, det_ckpt, device)
        self.tmm = TMMModule().to(device)
        self.motion = MotionPredictor(self.tmm.tim.encoder.layers[0].d_model).to(device)
        self.device = device
        self.load_model(model_ckpt)

    def load_model(self, ckpt):
        state = torch.load(ckpt)
        self.tmm.load_state_dict(state['model'])
        self.motion.load_state_dict(state.get('motion', {}))

    def track_sequence(self, frames):
        tracks = []
        memory = []
        for img in frames:
            dets = self.detector.detect(img)
            bboxes = dets[:, :4]
            scores = dets[:, 4]
            # update memory
            memory.append(bboxes)
            if len(memory) > 10:
                memory.pop(0)
            traj = torch.stack(memory, dim=1)  # [N,T,4]
            feat = self.tmm(traj.unsqueeze(0))    # [1,T,N,D]
            pred = self.motion(feat)[0]          # [N,4]
            # match
            cost = torch.cdist(pred, bboxes)
            row, col = linear_sum_assignment(cost.cpu().numpy())
            matched = list(zip(row, col))
            # update tracks
            for r,c in matched:
                tracks.append((r, c))
        return tracks