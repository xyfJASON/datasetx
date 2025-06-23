# Places365

[Official website](http://places2.csail.mit.edu/index.html) | [Papers with Code](https://paperswithcode.com/dataset/places365) | [Download](http://places2.csail.mit.edu/download-private.html)

## Brief introduction

> Copied from paperswithcode.

The Places365 dataset is a scene recognition dataset. It is composed of 10 million images comprising 434 scene classes. There are two versions of the dataset: Places365-Standard with 1.8 million train and 36000 validation images from K=365 scene classes, and Places365-Challenge-2016, in which the size of the training set is increased up to 6.2 million extra images, including 69 new scene classes (leading to a total of 8 million train images from 434 scene classes).

## Statistics

### Places365-Standard

**Numbers**: 2,168,460

**Splits**: 1,803,460 / 36,500 / 328,500 (train / valid / test)

**Resolution**:

- High-resolution: resized to have a minimum dimension of 512 while preserving the aspect ratio of the image. Original images that had a dimension smaller than 512 have been left unchanged.
- Small: resized to 256 * 256 regardless of the original aspect ratio.

**Annotations**: 365 scene categories

### Places365-Challenge-2016

**Notes** (copied from official website): Compared to the train set of Places365-Standard, the train set of Places365-Challenge has 6.2 million extra images, leading to totally 8 million train images for the Places365 challenge 2016. The validation set and testing set are the same as the Places365-Standard.

**Numbers**: 8,391,628

**Splits**: 8,026,628 / 36,500 / 328,500 (train / valid / test)

**Resolution**: The same as Places365-Standard.

**Annotations**: 365 scene categories

## Files

<table>
<tr><th colspan="3">Content</th><th>Filename</th><th>Size</th><th>MD5</th></tr>
<tr><td rowspan="4">High-res</td><td rowspan="2">Train</td><td>Standard</td><td>train_large_places365standard.tar</td><td>105GB</td><td>67e186b496a84c929568076ed01a8aa1</td></tr>
<tr><td>Challenge</td><td>train_large_places365challenge.tar</td><td>476GB</td><td>605f18e68e510c82b958664ea134545f</td></tr>
<tr><td colspan="2">Validation</td><td>val_large.tar</td><td>2.1GB</td><td>9b71c4993ad89d2d8bcbdc4aef38042f</td></tr>
<tr><td colspan="2">Test</td><td>test_large.tar</td><td>19GB</td><td>41a4b6b724b1d2cd862fb3871ed59913</td></tr>
<tr><td rowspan="4">Small</td><td rowspan="2">Train</td><td>Standard</td><td>train_256_places365standard.tar</td><td>24GB</td><td> 53ca1c756c3d1e7809517cc47c5561c5 </td></tr>
<tr><td>Challenge</td><td>train_256_places365challenge.tar</td><td> 108GB </td><td> 741915038a5e3471ec7332404dfb64ef </td></tr>
<tr><td colspan="2">Validation</td><td>val_256.tar</td><td> 501MB </td><td> e27b17d8d44f4af9a78502beb927f808 </td></tr>
<tr><td colspan="2">Test</td><td>test_256.tar</td><td> 4.4GB </td><td> f532f6ad7b582262a2ec8009075e186b </td></tr>
</table>

## Usage

### File structure

Please organize the downloaded dataset in the following file structure:

```text
root
├── categories_places365.txt
├── places365_train_standard.txt
├── places365_train_challenge.txt
├── places365_val.txt
├── places365_test.txt
├── data_256_standard                  (extracted from train_256_places365standard.tar)
│   ├── a
│   ├── ...
│   └── z
├── data_large_standard                (extracted from train_large_places365standard.tar)
│   ├── a
│   ├── ...
│   └── z
├── val_256                            (extracted from val_256.tar)
│   ├── Places365_val_00000001.jpg
│   ├── ...
│   └── Places365_val_00036500.jpg
├── val_large                          (extracted from val_large.tar)
│   ├── Places365_val_00000001.jpg
│   ├── ...
│   └── Places365_val_00036500.jpg
├── test_256                           (extracted from test_256.tar)
│   ├── Places365_test_00000001.jpg
│   ├── ...
│   └── Places365_test_00328500.jpg
└── test_large                         (extracted from test_large.tar)
    ├── Places365_test_00000001.jpg
    ├── ...
    └── Places365_test_00328500.jpg
```

### Example

```python
from image_datasets import Places365

root = '~/data/Places365'  # path to downloaded dataset
train_set = Places365(root=root, split='train', small=True)
valid_set = Places365(root=root, split='valid', small=True)
test_set = Places365(root=root, split='test', small=True)
print(len(train_set))  # 1803460
print(len(valid_set))  # 36500
print(len(test_set))   # 328500
print(train_set[0])    # (<PIL.Image.Image image mode=RGB size=256x256 at 0x7FD8EE031C10>, 0)
print(valid_set[100])  # (<PIL.Image.Image image mode=RGB size=256x256 at 0x7FCDF70A2E50>, 296)
print(test_set[1000])  # (<PIL.Image.Image image mode=RGB size=256x256 at 0x7FCDF70A2E50>, None)
```
