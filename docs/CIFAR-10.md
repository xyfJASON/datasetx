# CIFAR-10

[Official website](https://www.cs.toronto.edu/~kriz/cifar.html) | [Papers with Code](https://paperswithcode.com/dataset/cifar-10)

## Brief introduction

> Copied from paperswithcode.

The CIFAR-10 dataset (Canadian Institute for Advanced Research, 10 classes) is a subset of the Tiny Images dataset and consists of 60000 32x32 color images. The images are labelled with one of 10 mutually exclusive classes: airplane, automobile (but not truck or pickup truck), bird, cat, deer, dog, frog, horse, ship, and truck (but not pickup truck). There are 6000 images per class with 5000 training and 1000 testing images per class.

The criteria for deciding whether an image belongs to a class were as follows:

 - The class name should be high on the list of likely answers to the question “What is in this picture?”
 - The image should be photo-realistic. Labelers were instructed to reject line drawings.
 - The image should contain only one prominent instance of the object to which the class refers. The object may be partially occluded or seen from an unusual viewpoint as long as its identity is still clear to the labeler.

## Statistics

**Numbers**: 60,000

**Splits** (train / test): 50,000 / 10,000

**Resolution**: 32×32

**Labels**: 10 classes

## Usage

### Download

`torchvision` will automatically download the dataset with `download=True`.
Or you can also download the dataset with the following command:

```shell
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
md5sum cifar-10-python.tar.gz  # c58f30108f718f92721af3b95e74349a
tar -xzf cifar-10-python.tar.gz
```

### File structure

Please organize the dataset in the following file structure:

```text
root
├── cifar-10-python.tar.gz   (170.5 MB)
└── cifar-10-batches-py      (extracted from cifar-10-python.tar.gz)
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── readme.html
    └── test_batch
```

### Example

```python
from torchvision.datasets import CIFAR10

root = '~/data/CIFAR-10/'  # path to the dataset
train_set = CIFAR10(root=root, train=True)
test_set = CIFAR10(root=root, train=False)
print(len(train_set))  # 50000
print(len(test_set))   # 10000
print(train_set[0])    # (<PIL.Image.Image image mode=RGB size=32x32 at 0x7FC9A7FB7ED0>, 6)
```
