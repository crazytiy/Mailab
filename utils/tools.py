import os
import yaml
import random
from collections import namedtuple
from typing import Any, Dict, Union
import numpy as np
import torch
import pickle
from torchvision import transforms

def seed_everything(seed=666, workers= False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"


def  make_targz(output_filename, source_dir):
    import tarfile
    with tarfile. open (output_filename,  "w:gz" ) as tar:
        tar.add(source_dir, arcname = os.path.basename(source_dir))

def join(loader, node):
    seq = loader.construct_sequence(node)
    return os.path.join(*seq)

def todevice(loader, node):
    dev = loader.construct_sequence(node)
    if len(dev) != 1:
        raise ValueError(f'The length of device string must be is 1! {dev} is {len(dev)}.')

    return torch.device(dev[0])

## register the tag handler
yaml.add_constructor('!join', join)
yaml.add_constructor('!todevice', todevice)


def merge_dicts(configs):
    dicts = {}
    for i in getattr(configs, configs.scheduler):
        dicts.update(i)
    return dicts


def check_dir(path):
    """
    :param path(str): the path need to check whether exists in or not
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_config(config):
    cf = yaml.load(open(config), Loader=yaml.Loader)
    return namedtuple('config', cf.keys())(*cf.values())


def load_config(config: Union[str, Dict], inheritance_key: str = 'INHERIT') -> Dict[str, Any]:
    """Reads YAML configuration file with nested inheritance from other YAML files.
    Arguments:
        config {Union[str, Dict]} -- Configuration path/dictionary
    Keyword Arguments:
        inheritance_key {str} -- String used for inheritance paths (default: {'FROM'})
    Returns:
        Dict[str, Any] -- Configuration dictionary
    """
    if isinstance(config, str):
        config_dict = yaml.safe_load(open(config))
    elif isinstance(config, dict):
        config_dict = config
    else:
        raise ValueError(f'Expected config to be a str or dict but got {type(config)}.')

    if inheritance_key in config_dict:
        for yaml_file in config_dict[inheritance_key]:
            parent_config = load_config(yaml_file, inheritance_key)
            parent_config.update(config_dict)
            config_dict = parent_config

    return config_dict

def get_mean_std(dir,y_name='rain',vals_idx=None):
    '''
        返回mean_std和transform
    '''
    with open(os.path.join(dir,y_name+'.pkl'),'rb') as fy:
        ydicts=pickle.load(fy)
    # y_mean_std=np.array([ydicts['mean'],ydicts['std']])
    mean=ydicts['mean']
    std=ydicts['std']
    # y_mean_std=torch.from_numpy(y_mean_std).float()
    with open(os.path.join(dir,'x.pkl'),'rb') as fx:
        xdicts=pickle.load(fx)     
    x_mean_std=np.hstack((xdicts['mean'].reshape(-1,1),xdicts['std'].reshape(-1,1)))
    if vals_idx is not None:
        x_mean_std=x_mean_std[vals_idx,:]        
    transform =transforms.Normalize(
                mean=x_mean_std[:,0].tolist(),
                std=x_mean_std[:,1].tolist())
    return transform,mean,std

