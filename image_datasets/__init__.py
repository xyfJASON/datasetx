from .afhq import AFHQ
from .celeba import CelebA
from .celebahq import CelebAHQ
from .celebamask_hq import CelebAMaskHQ
from .cifar10 import CIFAR10
from .danbooru2019_portraits import Danbooru2019Portraits
from .ffhq import FFHQ
from .imagenet import ImageNet
from .mnist import MNIST
from .places365 import Places365
from .svhn import SVHN

__all__ = [
    'AFHQ',
    'CelebA',
    'CelebAHQ',
    'CelebAMaskHQ',
    'CIFAR10',
    'Danbooru2019Portraits',
    'FFHQ',
    'ImageNet',
    'MNIST',
    'Places365',
    'SVHN',
]
