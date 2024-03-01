from .UnetModel import Grid_Unet
from .unet_ddpm import UNet
from .inceptionnext_U import UPerNet
from .u2net_refactor import U2NET_lite,U2NET_full


def get_model(configs):
    model=None
    return model