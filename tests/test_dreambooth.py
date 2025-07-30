import unittest

import torch
import torchvision.transforms as T
from torch import Tensor

from datasetx import DreamBooth


class TestDreamBooth(unittest.TestCase):

    root = '~/data/dreambooth'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing DreamBooth dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size: int = None):
        # check sample, {image, prompt}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'prompt'})
        image, prompt = sample['image'], sample['prompt']
        # check image, squared
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        if size is not None:
            self.assertEqual(image.shape, (3, size, size))
        else:
            self.assertEqual(image.shape[0], 3)
            self.assertEqual(image.shape[1], image.shape[2])
        # check prompt, str
        self.assertIsInstance(prompt, str)

    def test_dataset(self):
        dataset = DreamBooth(self.root)
        self.assertEqual(len(dataset), 750)
        self.check_sample(dataset[0])

    def test_transform_fn(self):
        dataset = DreamBooth(
            root=self.root,
            transform_fn=DreamBoothTransform(size=512),
        )
        self.check_sample(dataset[0], size=512)


class DreamBoothTransform:
    def __init__(self, size: int = 512):
        self.image_transform = T.Compose([
            T.Resize((size, size), antialias=True),
            T.RandomHorizontalFlip(),
            T.Normalize(0.5, 0.5),
        ])

    def __call__(self, sample: dict):
        sample['image'] = self.image_transform(sample['image'])
        return sample
