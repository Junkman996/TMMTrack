import cv2
import numpy as np

def draw_tracks(frame, tracks, bboxes):
    for tid, idx in tracks:
        x1,y1,x2,y2 = bboxes[idx].tolist()
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame, str(tid), (int(x1),int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0),2)
    return frame