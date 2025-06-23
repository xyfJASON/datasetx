# CelebAMask-HQ

[Official website](https://mmlab.ie.cuhk.edu.hk/projects/CelebA/CelebAMask_HQ.html) | [Papers with Code](https://paperswithcode.com/dataset/celebamask-hq) | [Google drive](https://drive.google.com/open?id=1badu11NqxGf6qM3PTTooQDJvQbejgbTv) | [Baidu drive](https://pan.baidu.com/s/1wN1E-B1bJ7mE1mrn9loj5g)

## Brief introduction

> Copied from official website.

CelebAMask-HQ is a large-scale face image dataset that has 30,000 high-resolution face images selected from the CelebA dataset by following CelebA-HQ. Each image has segmentation mask of facial attributes corresponding to CelebA. The masks of CelebAMask-HQ were manually-annotated with the size of 512 x 512 and 19 classes including all facial components and accessories such as skin, nose, eyes, eyebrows, ears, mouth, lip, hair, hat, eyeglass, earring, necklace, neck, and cloth.

CelebAMask-HQ can be used to train and evaluate algorithms of face parsing, face recognition, and GANs for face generation and editing.

## Statistics

**Numbers**: 30,000 (same as CelebA-HQ dataset)

**Splits** (train / valid / test): 24,183 / 2,993 / 2,824 (following CelebA's original splits)

**Resolution**: 1024×1024 (images), 512x512 (masks)

**Attribute labels**: 40 binary labels (same as the original CelebA dataset)

**Masks**: 19 classes

**Pose annotations**: Yaw, Pitch and Raw for each image

## Usage

### Download

```shell
gdown 1badu11NqxGf6qM3PTTooQDJvQbejgbTv -O CelebAMask-HQ.zip
unzip CelebAMask-HQ.zip
```

### Generate masks

The authors provided black-white masks for each attribute under `CelebAMask-HQ-mask-anno`, and some [scripts](https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing) to generate index mask images (pixel value represents class label) and colorful mask images. I adapted the scripts to use multiprocessing and put it at [`scripts/celebamask_hq_generate_mask.py`](../scripts/celebamask_hq_generate_mask.py). The processed masks will be stored under `CelebAMask-HQ-mask` and `CelebAMask-HQ-mask-color`.

```shell
python celebamask_hq_generate_mask.py --root ROOT
```

### File structure

```text
root
├── CelebA-HQ-img
│   ├── 0.jpg
│   ├── ...
│   └── 29999.jpg
├── CelebA-HQ-to-CelebA-mapping.txt
├── CelebAMask-HQ-attribute-anno.txt
├── CelebAMask-HQ-mask-anno
├── CelebAMask-HQ-mask
│   ├── 0.png
│   ├── ...
│   └── 29999.png
├── CelebAMask-HQ-mask-color
│   ├── 0.png
│   ├── ...
│   └── 29999.png
├── CelebAMask-HQ-pose-anno.txt
└── README.txt
```

### Example

```python
from image_datasets import CelebAMaskHQ
from image_datasets.celebamask_hq import Compose, Resize, ToTensor, Normalize

root = '~/data/CelebAMask-HQ'  # path to the dataset

transforms = Compose([
    Resize((512, 512)),
    ToTensor(),
    Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

train_set = CelebAMaskHQ(root=root, split='train', transforms=transforms)
valid_set = CelebAMaskHQ(root=root, split='valid', transforms=transforms)
test_set = CelebAMaskHQ(root=root, split='test', transforms=transforms)
all_set = CelebAMaskHQ(root=root, split='all', transforms=transforms)

print(len(train_set))  # 24183
print(len(valid_set))  # 2993
print(len(test_set))   # 2824
print(len(all_set))    # 30000

print(train_set[0][0].shape, train_set[0][0].dtype)  # image: (3, 512, 512), torch.float32
print(train_set[0][1].shape, train_set[0][1].dtype)  # index mask: (512, 512), torch.int64
print(train_set[0][2].shape, train_set[0][2].dtype)  # color mask: (3, 512, 512), torch.float32
```
