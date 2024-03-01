from .dataset import *
from torch.utils.data import DataLoader

def get_loaders(*args,**kwargs):
    '''
    '''
    ds = AIData()

    loader =DataLoader(
        ds,
        # batch_size=batch_size,
        # num_workers=num_works,
        # pin_memory=pin_memory,
        # shuffle=shuffle,
    )
    return loader