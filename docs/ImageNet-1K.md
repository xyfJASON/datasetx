# ImageNet-1K (ILSVRC2012)

[Official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php)

## Brief introduction

The most commonly used ImageNet subset.

## Statistics

**Numbers**: 1,431,167

**Splits** (train / valid / test): 1,281,167 / 50,000 / 100,000

**Resolution**: inconsistent, with an average of ~469×387

**Annotations**: 1000 classes. The mapping from class id to label name can be found [here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a).

## Files

<table>
<tr><th colspan="2">Content</th><th>Filename</th><th>Size</th><th>MD5</th></tr>
<tr><td rowspan="2">Development Kit</td><td>Development Kit (Task 1 & 2)</td><td>ILSVRC2012_devkit_t12.tar.gz</td><td>2.5MB</td><td>-</td></tr>
<tr><td>Development Kit (Task 3)</td><td>ILSVRC2012_devkit_t3.tar.gz</td><td>22MB</td><td>-</td></tr>
<tr><td rowspan="4">Images</td><td>Training images (Task 1 & 2)</td><td>ILSVRC2012_img_train.tar</td><td>138GB</td><td>1d675b47d978889d74fa0da5fadfb00e</td></tr>
<tr><td>Training images (Task 3)</td><td>ILSVRC2012_img_train_t3.tar</td><td>728MB</td><td>ccaf1013018ac1037801578038d370da</td></tr>
<tr><td>Validation images (all tasks)</td><td>ILSVRC2012_img_val.tar</td><td>6.3GB</td><td>29b22e2961454d5413ddabcf34fc5622</td></tr>
<tr><td>Test images (all tasks)</td><td>ILSVRC2012_img_test_v10102019.tar</td><td>13GB</td><td>e1b8681fff3d63731c599df9b4b6fc02</td></tr>
<tr><td rowspan="4">Bounding Boxes</td><td>Training bounding box annotations (Task 1 & 2 only)</td><td>ILSVRC2012_bbox_train_v2.tar.gz</td><td>20MB</td><td>9271167e2176350e65cfe4e546f14b17</td></tr>
<tr><td>Training bounding box annotations (Task 3 only)</td><td>ILSVRC2012_bbox_train_dogs.tar.gz</td><td>1MB</td><td>61ebd3cc0e4793899a841b6b27f3d6d8</td></tr>
<tr><td>Validation bounding box annotations (all tasks)</td><td>ILSVRC2012_bbox_val_v3.tgz</td><td>2.2MB</td><td>f4cd18b5ea29fe6bbea62ec9c20d80f0</td></tr>
<tr><td>Test bounding box annotations (Task 3 only)</td><td>ILSVRC2012_bbox_test_dogs.zip</td><td>33MB</td><td>2dfdb2677fd9661585d17d5a5d027624</td></tr>
</table>

## Usage

### Download

```shell
mkdir Images
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar -O Images/ILSVRC2012_img_train.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train_t3.tar -O Images/ILSVRC2012_img_train_t3.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar -O Images/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar -O Images/ILSVRC2012_img_test_v10102019.tar

mkdir Development_Kit
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz -O Development_Kit/ILSVRC2012_devkit_t12.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t3.tar.gz -O Development_Kit/ILSVRC2012_devkit_t3.tar.gz

mkdir Bounding_Boxes
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_v2.tar.gz -O Bounding_Boxes/ILSVRC2012_bbox_train_v2.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_train_dogs.tar.gz -O Bounding_Boxes/ILSVRC2012_bbox_train_dogs.tar.gz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz -O Bounding_Boxes/ILSVRC2012_bbox_val_v3.tgz
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_test_dogs.zip -O Bounding_Boxes/ILSVRC2012_bbox_test_dogs.zip
```

### File structure

Please organize the downloaded dataset in the following file structure:

```text
root
├── train
│   ├── n01440764
│   ├── ...
│   └── n15075141
├── val
│   ├── n01440764
│   ├── ...
│   └── n15075141
└── test
    ├── ILSVRC2012_test_00000001.JPEG
    ├── ...
    └── ILSVRC2012_test_00100000.JPEG
```

You can use the following script to extract the files:

```shell
mkdir train && tar -xvf ILSVRC2012_img_train.tar -C ./train
mkdir val && tar -xvf ILSVRC2012_img_val.tar -C ./val
tar -xvf ILSVRC2012_img_test_v10102019.tar

cd train
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

cd val
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

### Example

```python
from image_datasets import ImageNet

root = '~/data/ImageNet/ILSVRC2012/Images'  # path to downloaded dataset
train_set = ImageNet(root=root, split='train')
valid_set = ImageNet(root=root, split='valid')
test_set = ImageNet(root=root, split='test')
print(len(train_set))  # 1281167
print(len(valid_set))  # 50000
print(len(test_set))   # 100000
print(train_set[0])    # (<PIL.Image.Image image mode=RGB size=250x250 at 0x7F6263A32050>, 0)
print(valid_set[100])  # (<PIL.Image.Image image mode=RGB size=750x550 at 0x7F638B0E3CD0>, 2)
print(test_set[1000])  # (<PIL.Image.Image image mode=RGB size=500x391 at 0x7F6263A32050>, None)
```