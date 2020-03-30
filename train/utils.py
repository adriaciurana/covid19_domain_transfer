import torch
from collections import deque
import numpy as np
import copy
from collections import OrderedDict, Callable

# Utils
class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, max_len=3000):
        self.reset()
        self.val_deque = deque(maxlen=max_len)
        self.avg_deque = 0.0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if np.isinf(np.abs(val)) or np.isnan(val):
            return
        self.val_deque.append(val)
        self.avg_deque = np.mean(self.val_deque)

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
