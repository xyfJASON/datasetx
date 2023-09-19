## image-datasets

This package extends `torchvision.datasets` in the following ways:
 - Adds some commonly used datasets that are not available in `torchvision.datasets`, such as [FFHQ](https://github.com/NVlabs/ffhq-dataset).
 - For datasets already existing in `torchvision.datasets`, adds a wrapper class with pre-defined data transforms. The transform can be selected by a string argument `transform_type`. This helps to instantiate datasets with configuration files.

For now, the following datasets are supported:
 - [MNIST](http://yann.lecun.com/exdb/mnist/)
 - [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
 - [FFHQ](https://github.com/NVlabs/ffhq-dataset)
 - [AFHQ](https://github.com/clovaai/stargan-v2)
