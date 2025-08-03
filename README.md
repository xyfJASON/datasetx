<h1 align="center">datasetx</h1>

This package implements commonly used datasets based on `torch` and `torchvision`.

The API mostly follows `torchvision.datasets`, with several key differences:
 - Samples are returned as dictionaries rather than tuples, making them easier to understand.
 - Transform functions also take dictionaries as input, providing greater flexibility.
 - Images are represented as torch Tensors in the \[0,1\] range by default, instead of PIL Images.

## Installation

```shell
pip install git+https://github.com/xyfJASON/datasetx.git
```

## Documentation

Please see [docs](./docs) for detailed instructions on each dataset.
