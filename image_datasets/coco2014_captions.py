import os
import json
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class COCO2014Captions(Dataset):
    """COCO 2014 caption dataset with Karpathy split.

    Please organize the dataset in the following file structure:

    root
    ├── train2014
    │   ├── COCO_train2014_000000000009.jpg
    │   ├── ...
    │   └── COCO_train2014_000000581921.jpg
    ├── val2014
    │   ├── COCO_val2014_000000000042.jpg
    │   ├── ...
    │   └── COCO_val2014_000000581929.jpg
    ├── test2014
    │   ├── COCO_test2014_000000000001.jpg
    │   ├── ...
    │   └── COCO_test2014_000000581923.jpg
    ├── annotations
    │   ├── captions_train2014.json
    │   ├── captions_val2014.json
    │   ├── instances_train2014.json
    │   ├── instances_val2014.json
    │   ├── person_keypoints_train2014.json
    │   ├── person_keypoints_val2014.json
    │   └── image_info_test2014.json
    └── karpathy
        ├── dataset_coco.json
        ├── dataset_flickr8k.json  # not used
        └── dataset_flickr30k.json # not used

    Reference:
      - https://cocodataset.org
      - https://www.v7labs.com/blog/coco-dataset-guide
      - https://cs.stanford.edu/people/karpathy/deepimagesent

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform_fn: Optional[Callable] = None,
    ):
        self.root = os.path.expanduser(root)
        self.split = 'val' if split == 'valid' else split
        self.transform_fn = transform_fn

        with open(os.path.join(self.root, 'karpathy', 'dataset_coco.json'), 'r') as f:
            self.metadata = json.load(f)['images']
        if self.split == 'train':
            self.metadata = [data for data in self.metadata if data['split'] in ['train', 'restval']]
        elif self.split == 'val':
            self.metadata = [data for data in self.metadata if data['split'] == 'val']
        elif self.split == 'test':
            self.metadata = [data for data in self.metadata if data['split'] == 'test']
        else:
            raise ValueError(f'Unknown split {self.split}')

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        # read image and captions
        metadata = self.metadata[index]
        image_path = os.path.join(self.root, metadata['filepath'], metadata['filename'])
        image = Image.open(image_path).convert('RGB')
        captions = [caption['raw'] for caption in metadata['sentences']]
        # convert image to tensor
        image = to_tensor(image)
        sample = {'image': image, 'captions': captions}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
