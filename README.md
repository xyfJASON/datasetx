## image-datasets

This package extends `torchvision.datasets` in the following ways:
 - Adds some commonly used datasets that are not available in `torchvision.datasets`, such as FFHQ.
 - For datasets already existing in `torchvision.datasets`, adds a wrapper class with pre-defined data transforms. The transform can be selected by a string argument `transform_type`. This helps to instantiate datasets with configuration files.

For now, the following datasets are supported:
 - [AFHQ](https://github.com/clovaai/stargan-v2)
 - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
 - [CelebA-HQ](https://github.com/tkarras/progressive_growing_of_gans)
 - [CelebAMask-HQ](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html)
 - [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
 - [Danbooru2019 Portraits](https://gwern.net/crop#danbooru2019-portraits)
 - [FFHQ](https://github.com/NVlabs/ffhq-dataset)
 - [ImageNet (ILSVRC2012)](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)
 - [MNIST](http://yann.lecun.com/exdb/mnist/)
 - [SVHN](http://ufldl.stanford.edu/housenumbers/)

Please see [Wiki](https://github.com/xyfJASON/image-datasets/wiki) page for instructions on how to load them with `image-datasets`.
