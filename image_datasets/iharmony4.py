import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class iHarmony4(Dataset):
    """The iHarmony4 Dataset.

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
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')
        if subset not in [None, 'HCOCO', 'HAdobe5k', 'HFlickr', 'Hday2night']:
            raise ValueError(f'Invalid subset: {subset}')
        self.root = os.path.expanduser(root)
        self.split = split
        self.subset = subset
        self.transform_fn = transform_fn

        # read filenames from txt file
        txt_file = os.path.join(self.root, f'IHD_{self.split}.txt')
        if self.subset is not None:
            txt_file = os.path.join(self.root, self.subset, f'{self.subset}_{self.split}.txt')
        with open(txt_file, 'r') as f:
            filenames = f.readlines()

        # parse filenames to get image paths
        self.comp_image_paths, self.mask_paths, self.real_image_paths = self._parse_filenames(filenames)

    def __len__(self):
        return len(self.comp_image_paths)

    def __getitem__(self, index: int):
        # read image, mask, and real image
        comp_image = Image.open(self.comp_image_paths[index]).convert('RGB')
        mask = Image.open(self.mask_paths[index]).convert('L')
        real_image = Image.open(self.real_image_paths[index]).convert('RGB')
        # convert to tensor
        comp_image = to_tensor(comp_image)
        mask = to_tensor(mask) > 0.5
        real_image = to_tensor(real_image)
        sample = {'comp_image': comp_image, 'mask': mask, 'real_image': real_image}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample

    def _parse_filenames(self, filenames):
        if self.subset is None:
            dirs = [f.split('/')[0] for f in filenames]
            filenames = [f.split('/')[-1] for f in filenames]
        else:
            dirs = [self.subset] * len(filenames)

        comp_image_names, mask_names, real_image_names = [], [], []
        for f in filenames:
            f = os.path.splitext(f)[0]
            name_parts = f.split('_')
            comp_image_names.append(f'{f}.jpg')
            mask_names.append(f'{name_parts[0]}_{name_parts[1]}.png')
            real_image_names.append(f'{name_parts[0]}.jpg')

        comp_image_paths = [os.path.join(self.root, d, 'composite_images', n) for d, n in zip(dirs, comp_image_names)]
        mask_paths = [os.path.join(self.root, d, 'masks', n) for d, n in zip(dirs, mask_names)]
        real_image_paths = [os.path.join(self.root, d, 'real_images', n) for d, n in zip(dirs, real_image_names)]
        return comp_image_paths, mask_paths, real_image_paths
