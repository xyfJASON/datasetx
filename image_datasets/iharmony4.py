import os
import random
from PIL import Image
from typing import Optional, Callable

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets import VisionDataset


class iHarmony4(VisionDataset):
    """The iHarmony4 Dataset.

    iHarmony4 is a synthesized dataset for Image Harmonization. It contains 4 sub-datasets: HCOCO, HAdobe5k, HFlickr,
    and Hday2night (based on COCO, Adobe5k, Flickr, day2night datasets respectively), each of which contains
    synthesized composite images, foreground masks of composite images and corresponding real images.
    (Copied from PapersWithCode)

    Please organize the dataset in the following file structure:

    root
    ├── IHD_train.txt
    ├── IHD_test.txt
    ├── HAdobe5k
    │   ├── HAdobe5k_train.txt
    │   ├── HAdobe5k_test.txt
    │   ├── composite_images
    │   ├── masks
    │   └── real_images
    ├── HCOCO
    │   ├── HCOCO_train.txt
    │   ├── HCOCO_test.txt
    │   ├── composite_images
    │   ├── masks
    │   └── real_images
    ├── Hday2night
    │   ├── Hday2night_train.txt
    │   ├── Hday2night_test.txt
    │   ├── composite_images
    │   ├── masks
    │   └── real_images
    └── HFlickr
        ├── HFlickr_train.txt
        ├── HFlickr_test.txt
        ├── composite_images
        ├── masks
        └── real_images

    Reference:
     - https://github.com/bcmi/Image-Harmonization-Dataset-iHarmony4
     - https://paperswithcode.com/dataset/iharmony4

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            subset: str = None,  # None for all subsets
            transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)

        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        self.split = split

        if subset not in [None, 'HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']:
            raise ValueError(f'Invalid subset: {subset}')
        self.subset = subset

        # Read filenames from txt file
        txt_file = os.path.join(self.root, f'IHD_{self.split}.txt')
        if self.subset is not None:
            txt_file = os.path.join(self.root, self.subset, f'{self.subset}_{self.split}.txt')
        with open(txt_file, 'r') as f:
            filenames = f.readlines()

        # Parse filenames to get image paths
        self.comp_img_paths, self.mask_paths, self.real_img_paths = self._parse_filenames(filenames)

    def __len__(self):
        return len(self.comp_img_paths)

    def __getitem__(self, index: int):
        comp_img = Image.open(self.comp_img_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('L')
        real_img = Image.open(self.real_img_paths[index]).convert('RGB')
        if self.transforms is not None:
            comp_img, mask, real_img = self.transforms(comp_img, mask, real_img)
        return comp_img, mask, real_img

    def _parse_filenames(self, filenames):
        if self.subset is None:
            dirs = [f.split('/')[0] for f in filenames]
            filenames = [f.split('/')[-1] for f in filenames]
        else:
            dirs = [self.subset] * len(filenames)

        comp_img_names, mask_names, real_img_names = [], [], []
        for f in filenames:
            f = os.path.splitext(f)[0]
            name_parts = f.split('_')
            comp_img_names.append(f'{f}.jpg')
            mask_names.append(f'{name_parts[0]}_{name_parts[1]}.png')
            real_img_names.append(f'{name_parts[0]}.jpg')

        comp_img_paths = [os.path.join(self.root, d, 'composite_images', n) for d, n in zip(dirs, comp_img_names)]
        mask_paths = [os.path.join(self.root, d, 'masks', n) for d, n in zip(dirs, mask_names)]
        real_img_paths = [os.path.join(self.root, d, 'real_images', n) for d, n in zip(dirs, real_img_names)]
        return comp_img_paths, mask_paths, real_img_paths


# ===============================================================================================
# Below are custom transforms that apply to composite_image, mask and real_image simultaneously
# Adapted from https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py
# ===============================================================================================

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = TF.pad(img, [0, 0, padw, padh], fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, comp_img, mask, real_img):
        for t in self.transforms:
            comp_img, mask, real_img = t(comp_img, mask, real_img)
        return comp_img, mask, real_img


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, comp_img, mask, real_img):
        comp_img = TF.resize(comp_img, self.size, antialias=True)
        mask = TF.resize(mask, self.size, antialias=True)
        real_img = TF.resize(real_img, self.size, antialias=True)
        return comp_img, mask, real_img


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, comp_img, mask, real_img):
        comp_img = TF.center_crop(comp_img, self.size)
        mask = TF.center_crop(mask, self.size)
        real_img = TF.center_crop(real_img, self.size)
        return comp_img, mask, real_img


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, comp_img, mask, real_img):
        comp_img = pad_if_smaller(comp_img, self.size)
        mask = pad_if_smaller(mask, self.size)
        real_img = pad_if_smaller(real_img, self.size)
        crop_params = T.RandomCrop.get_params(comp_img, (self.size, self.size))
        comp_img = TF.crop(comp_img, *crop_params)
        mask = TF.crop(mask, *crop_params)
        real_img = TF.crop(real_img, *crop_params)
        return comp_img, mask, real_img


class RandomHorizontalFlip:
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, comp_img, mask, real_img):
        if random.random() < self.flip_prob:
            comp_img = TF.hflip(comp_img)
            mask = TF.hflip(mask)
            real_img = TF.hflip(real_img)
        return comp_img, mask, real_img


class ToTensor:
    def __call__(self, comp_img, mask, real_img):
        comp_img = TF.to_tensor(comp_img)
        mask = TF.to_tensor(mask)
        real_img = TF.to_tensor(real_img)
        return comp_img, mask, real_img


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, comp_img, mask, real_img):
        comp_img = TF.normalize(comp_img, mean=self.mean, std=self.std)
        real_img = TF.normalize(real_img, mean=self.mean, std=self.std)
        return comp_img, mask, real_img
