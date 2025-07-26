import random
import unittest

import torch
import torchvision.transforms as T
from torch import Tensor

from datasetx import COCO2014Captions


class TestCOCO2014Captions(unittest.TestCase):

    root = '~/data/COCO2014'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing COCO2014 Captions dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size: int = None):
        # check sample, {image, captions}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'captions'})
        image, captions = sample['image'], sample['captions']
        # check image
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        if size is not None:
            self.assertEqual(image.shape, (3, size, size))
        # check captions
        self.assertIsInstance(captions, list)
        self.assertEqual(len(captions), 5)
        for caption in captions:
            self.assertIsInstance(caption, str)

    def test_train_split(self):
        train_set = COCO2014Captions(self.root, split='train')
        self.assertEqual(len(train_set), 113287)
        self.check_sample(train_set[0])

    def test_valid_split(self):
        valid_set = COCO2014Captions(self.root, split='valid')
        self.assertEqual(len(valid_set), 5000)
        self.check_sample(valid_set[0])

    def test_test_split(self):
        test_set = COCO2014Captions(self.root, split='test')
        self.assertEqual(len(test_set), 5000)
        self.check_sample(test_set[0])

    def test_transform_fn(self):
        train_set = COCO2014Captions(
            root=self.root,
            split='train',
            transform_fn=COCO2014CaptionsTransform(size=512),
        )
        sample = train_set[0]
        self.assertEqual(sample.keys(), {'image', 'caption'})
        self.assertEqual(sample['image'].shape, (3, 512, 512))
        self.assertIsInstance(sample['caption'], str)


class COCO2014CaptionsTransform:
    def __init__(self, size):
        self.image_transform = T.Compose([
            T.Resize(size, antialias=True),
            T.CenterCrop(size),
            T.Normalize(mean=0.5, std=0.5),
        ])

    def __call__(self, sample: dict):
        image, captions = sample['image'], sample['captions']
        # image transform
        image = self.image_transform(image)
        # random pick one caption
        caption = random.choice(captions)
        sample = {'image': image, 'caption': caption}
        return sample
