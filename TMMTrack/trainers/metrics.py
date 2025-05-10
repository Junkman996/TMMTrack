import numpy as np
from motmetrics import MetricsHost, io as mot_io

class MetricLogger:
    def __init__(self):
        self.metrics = []
        self.host = MetricsHost()

    def update(self, gt_ids, pred_ids, distances):
        # distances: cost matrix
        acc = self.host.update(gt_ids, pred_ids, distances)
        self.metrics.append(acc)

    def summarize(self):
        mh = mot_io.render_summary(self.host.compute(), formatters=mot_io.mh_formatters, namemap=mot_io.mh_names)
        return mh