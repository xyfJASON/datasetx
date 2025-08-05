import os
from PIL import Image
from pathlib import Path
from typing import Optional, Callable

import torchaudio
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class SoundNet(Dataset):
    """The SoundNet Dataset.

    Please organize the dataset in the following file structure:

    root
    ├── lists
    │   ├── train_frames4_0001.txt
    │   ├── ...
    │   ├── train_frames4_0040.txt
    │   ├── val_frames4_0001.txt
    │   ├── ...
    │   └── val_frames4_0010.txt
    ├── frames
    │   ├── videos
    │   └── videos2
    └── mp3
        ├── videos
        └── videos2

    References:
      - https://soundnet.csail.mit.edu/

    """

    def __init__(
            self,
            root: str,
            split: str = 'train',
            transform_fn: Optional[Callable] = None,
    ):
        if split not in ['train', 'val', 'valid']:
            raise ValueError(f'Invalid split: {split}')
        self.root = os.path.expanduser(root)
        self.split = split if split != 'valid' else 'val'
        self.transform_fn = transform_fn

        self.file_paths = []
        for txtfile in Path(os.path.join(self.root, 'lists')).glob(f'{self.split}_frames*.txt'):
            with open(txtfile, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.split(' ')[0].split('frames/')[1].strip()
                image_path = os.path.join(self.root, 'frames', line)
                audio_path = os.path.join(self.root, 'mp3', '/'.join(line.split('/')[:-1]) + '.mp3')
                self.file_paths.append((image_path, audio_path))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index: int):
        image_path, audio_path = self.file_paths[index]
        # read image
        image = Image.open(image_path).convert('RGB')
        image = to_tensor(image)
        # read audio
        audio, audio_sample_rate = torchaudio.load(audio_path)
        audio = audio.mean(dim=0)  # convert to mono
        sample = {'image': image, 'audio': audio, 'audio_sample_rate': audio_sample_rate}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
