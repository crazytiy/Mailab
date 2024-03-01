import os
import numpy
import torch
from torch.utils.data.dataset import Dataset

def load_dataset(*args,**kwargs):
    ds = []
    return ds

class AIData(Dataset):
    def __init__(self, *args,**kwargs):
        super(AIData, self).__init__()
        self.ds=load_dataset()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        input,target=self.ds[idx]
        return input, target