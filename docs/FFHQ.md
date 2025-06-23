# FFHQ

[Official website](https://github.com/NVlabs/ffhq-dataset) | [Papers with Code](https://paperswithcode.com/dataset/ffhq) | [Google drive](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP)

## Brief introduction

> Copied from paperswithcode.

Flickr-Faces-HQ (FFHQ) consists of 70,000 high-quality PNG images at 1024×1024 resolution and contains considerable variation in terms of age, ethnicity and image background. It also has good coverage of accessories such as eyeglasses, sunglasses, hats, etc. The images were crawled from Flickr, thus inheriting all the biases of that website, and automatically aligned and cropped using dlib. Only images under permissive licenses were collected. Various automatic filters were used to prune the set, and finally Amazon Mechanical Turk was used to remove the occasional statues, paintings, or photos of photos.

## Statistics

**Numbers**: 70,000

**Splits** (train / test): 60,000 / 10,000

**Resolution**: 1024×1024

## Files

<table>
<tr><th>Content</th><th>Filename</th><th>Size</th></tr>
<tr><td>Metadata</td><td>ffhq-dataset-v2.json</td><td>255 MB</td></tr>
<tr><td>Aligned and cropped images at 1024×1024</td><td>images1024x1024.zip</td><td>89.1 GB</td></tr>
<tr><td>Thumbnails at 128×128</td><td>thumbnails128x128.zip</td><td>1.95 GB</td></tr>
<tr><td>Original images from Flickr</td><td>in-the-wild-images.zip</td><td>955 GB</td></tr>
<tr><td>Multi-resolution data for StyleGAN and StyleGAN2</td><td>tfrecords.zip</td><td>273 GB</td></tr>
</table>

## Usage

> The authors provide several versions of dataset (see the table [here](https://github.com/NVlabs/ffhq-dataset#overview)), among which `images1024x1024` is the most commonly used one. This package supports `images1024x1024` and `thumbnails128x128` for now.

### Download

```shell
gdown <file id> -O <filename>
```

| filename              | file id                           | md5sum                           |
|-----------------------|-----------------------------------|----------------------------------|
| ffhq-dataset-v2.json  | 16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA | 425ae20f06a4da1d4dc0f46d40ba5fd6 |
| LICENSE.txt           | 1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX | 724f3831aaecd61a84fe98500079abc2 |
| README.txt            | 1UZXmHOdp2RLR4RKctn1msY5Eg4mXU3Q6 | -                                |
| images1024x1024.zip   | 1WvlAIvuochQn_L_f9p3OdFdTiSLlnnhv | -                                |
| thumbnails128x128.zip | 1Wrr6qZA1Tr6r9edNL2nSxnwopMW1n6pR | -                                |

### File structure

Please organize the downloaded dataset in the following file structure:

```text
root
├── ffhq-dataset-v2.json
├── LICENSE.txt
├── README.txt
├── thumbnails128x128
│   ├── 00000.png
│   ├── ...
│   └── 69999.png
└── images1024x1024
    ├── 00000.png
    ├── ...
    └── 69999.png
```

### Example

```python
from image_datasets import FFHQ

root = '~/data/FFHQ'  # path to the dataset
train_set = FFHQ(root=root, split='train')
test_set = FFHQ(root=root, split='test')
all_set = FFHQ(root=root, split='all')
print(len(train_set))  # 60000
print(len(test_set))   # 10000
print(len(all_set))    # 70000
print(train_set[0])    # <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1024x1024 at 0x7FD1B8BF1A50>
```
