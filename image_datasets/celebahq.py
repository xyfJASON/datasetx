import os
import shutil
from PIL import Image
from typing import Optional, Callable

import torchvision.transforms as T
from torch.utils.data import Dataset

from .utils import extract_images


class CelebAHQ(Dataset):
    """The CelebA-HQ Dataset.

    The CelebA-HQ dataset is a high-quality version of CelebA that consists of 30,000 images at 1024×1024 resolution.
    (Copied from PaperWithCode)

    The official way to prepare the dataset is to download img_celeba.7z from the original CelebA dataset and the delta
    files from the official GitHub repository. Then use dataset_tool.py to generate the high-quality images.

    However, I personally recommend downloading the CelebAMask-HQ dataset, which contains processed CelebA-HQ images.
    The file names in CelebAMask-HQ are sorted from 0 to 29999, which is inconsistent with the original CelebA file
    names. This class will automatically check and convert the file names to the original CelebA file names.

    To load data with this class, the dataset should be organized in the following structure:

    root
    ├── CelebA-HQ-img
    │   ├── 000004.jpg
    │   ├── ...
    │   └── 202591.jpg
    └── CelebA-HQ-to-CelebA-mapping.txt

    The train/valid/test sets are split according to the original CelebA dataset,
    resulting in 24,183 training images, 2,993 validation images, and 2,824 test images.

    This class has one pre-defined transform:
      - 'resize' (default): Resize the image directly to the target size

    References:
      - https://github.com/tkarras/progressive_growing_of_gans
      - https://paperswithcode.com/dataset/celeba-hq
      - https://github.com/switchablenorms/CelebAMask-HQ

    """
    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
            transform_type: str = 'default',
            transform: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        root = os.path.expanduser(root)
        image_root = os.path.join(root, 'CelebA-HQ-img')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')

        self.root = root
        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        self.transform = transform
        if transform is None:
            self.transform = self.get_transform()

        self.img_paths = extract_images(image_root)
        self.map_filenames()

        def filter_func(p):
            if split == 'all':
                return True
            celeba_splits = [1, 162771, 182638, 202600]
            k = 0 if split == 'train' else (1 if split == 'valid' else 2)
            return celeba_splits[k] <= int(os.path.splitext(os.path.basename(p))[0]) < celeba_splits[k+1]

        self.img_paths = list(filter(filter_func, self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item])
        if self.transform is not None:
            X = self.transform(X)
        return X

    def get_transform(self):
        flip_p = 0.5 if self.split in ['train', 'all'] else 0.0
        if self.transform_type in ['default', 'resize']:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none':
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform

    def map_filenames(self):
        filenames = [int(os.path.splitext(os.path.basename(p))[0]) for p in self.img_paths]
        if list(sorted(filenames)) == list(range(30000)):
            print('Mapping CelebAMask-HQ filenames to CelebA filenames...')
            import pandas as pd
            mapping = pd.read_table(os.path.join(self.root, 'CelebA-HQ-to-CelebA-mapping.txt'), sep=r'\s+', index_col=0)
            mapping_dict = {f'{i}.jpg': mapping.iloc[i]['orig_file'] for i in range(30000)}
            # Create a new directory `root/CelebA-HQ-img-tmp` to store the renamed images
            os.makedirs(os.path.join(self.root, 'CelebA-HQ-img-tmp'), exist_ok=True)
            for key, value in mapping_dict.items():
                if not os.path.isfile(os.path.join(self.root, 'CelebA-HQ-img', key)):
                    # If file not found, remove the temporary directory and raise an error
                    shutil.rmtree(os.path.join(self.root, 'CelebA-HQ-img-tmp'))
                    raise ValueError(f"{os.path.join(self.root, 'CelebA-HQ-img', key)} does not exist")
                # Copy the file to the temporary directory
                shutil.copy(os.path.join(self.root, 'CelebA-HQ-img', key),
                            os.path.join(self.root, 'CelebA-HQ-img-tmp', value))
            # Backup the original directory and rename the temporary directory
            shutil.move(os.path.join(self.root, 'CelebA-HQ-img'), os.path.join(self.root, 'CelebA-HQ-img-backup'))
            shutil.move(os.path.join(self.root, 'CelebA-HQ-img-tmp'), os.path.join(self.root, 'CelebA-HQ-img'))
            self.img_paths = extract_images(os.path.join(self.root, 'CelebA-HQ-img'))


if __name__ == '__main__':
    dataset = CelebAHQ(root='~/data/CelebA-HQ/', img_size=256, split='train')
    print(len(dataset))
    dataset = CelebAHQ(root='~/data/CelebA-HQ/', img_size=256, split='valid')
    print(len(dataset))
    dataset = CelebAHQ(root='~/data/CelebA-HQ/', img_size=256, split='test')
    print(len(dataset))
    dataset = CelebAHQ(root='~/data/CelebA-HQ/', img_size=256, split='all')
    print(len(dataset))
