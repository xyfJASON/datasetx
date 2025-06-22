# MNIST

[Official website](http://yann.lecun.com/exdb/mnist/) | [Papers with Code](https://paperswithcode.com/dataset/mnist)

## Brief introduction

> Copied from paperswithcode.

The MNIST database (Modified National Institute of Standards and Technology database) is a large collection of handwritten digits. It has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger NIST Special Database 3 (digits written by employees of the United States Census Bureau) and Special Database 1 (digits written by high school students) which contain monochrome images of handwritten digits. The digits have been size-normalized and centered in a fixed-size image. The original black and white (bilevel) images from NIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

## Statistics

**Numbers**: 70,000

**Splits** (train / test): 60,000 / 10,000

**Resolution**: 28×28

**Labels**：10 classes

## Usage

### Download

`torchvision` will automatically download the dataset with `download=True`.
Or you can also download the dataset with the following command:

```shell
mkdir -p MNIST/raw
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz -O MNIST/raw/train-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz -O MNIST/raw/train-labels-idx1-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz -O MNIST/raw/t10k-images-idx3-ubyte.gz
wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz -O MNIST/raw/t10k-labels-idx1-ubyte.gz
md5sum MNIST/raw/train-images-idx3-ubyte.gz  # f68b3c2dcbeaaa9fbdd348bbdeb94873
md5sum MNIST/raw/train-labels-idx1-ubyte.gz  # d53e105ee54ea40749a09fcbcd1e9432
md5sum MNIST/raw/t10k-images-idx3-ubyte.gz   # 9fb629c4189551a2d022fa330f9573f3
md5sum MNIST/raw/t10k-labels-idx1-ubyte.gz   # ec29112dd5afa0611ce80d1b7f02629c
```

### File structure

Please organize the dataset in the following file structure:

```text
root
└── MNIST
    └── raw
        ├── t10k-images-idx3-ubyte
        ├── t10k-images-idx3-ubyte.gz
        ├── t10k-labels-idx1-ubyte
        ├── t10k-labels-idx1-ubyte.gz
        ├── train-images-idx3-ubyte
        ├── train-images-idx3-ubyte.gz
        ├── train-labels-idx1-ubyte
        └── train-labels-idx1-ubyte.gz
```

### Example

```python
from torchvision.datasets import MNIST
train_set = MNIST(root='~/data/MNIST', train=True)
test_set = MNIST(root='~/data/MNIST', train=False)
print(len(train_set))  # 60000
print(len(test_set))   # 10000
```
