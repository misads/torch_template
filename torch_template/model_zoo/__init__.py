from .FFA import FFA
from .linknet import LinkNet
from .linknet import LinkNet50
from .unet import NestedUNet, UNet
from .pix2pix import GlobalGenerator, LocalEnhancer
from .transform_net import TransformNet
from .dense import Dense

model_zoo = {
    'FFA': FFA,
    'LinkNet': LinkNet,
    'LinkNet50': LinkNet50,
    'NestedUNet': NestedUNet,
    'UNet': UNet,
    'GlobalGenerator': GlobalGenerator,
    'LocalEnhancer': LocalEnhancer,
    'TransformNet': TransformNet,
    'Dense': Dense,
}

"""
  Model:      nf      n_params      256       512       256_batch8
   UNet        8       852,304       -        569M
NestedUNet    64     36,629,763    1851M     5365M        10037M
   FFA         -      4,455,913    5509M  out of memory out of memory
LinkNet50      -     28,762,115    1051M     1357M
 LinkNet       -     1,533,635      761M     1883M
  Global      64     45,614,595    1415M     1767M         2457M
TransformNet  32      1,673,097     829M     1587M         2615M
   Dense       ?     11,581,967     907M     1659M         2695M
   
"""