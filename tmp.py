import numpy as np
import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

st = np.load("st_Q_result.npy", allow_pickle='True').item()
#tc = np.load("tc_result.npy", allow_pickle='True').item()

for i in range(12):
    for j in range(12):
      avg = st[f"L{i}H{j}"][0].avg.item()
      print(f"L{i}H{j} : {avg}")
    print()