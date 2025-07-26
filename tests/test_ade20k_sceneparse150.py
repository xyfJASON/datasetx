import random
import unittest

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch import Tensor

from datasetx import ADE20KSceneParse150


class TestADE20KSceneParse150(unittest.TestCase):

    root = '~/data/ADE20K/SceneParsing'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing ADE20K SceneParse150 dataset...' + '\033[0m')

    def check_sample(self, sample: dict, size: int = None, anno_is_none: bool = False):
        # check sample, {image, annotation}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'annotation'})
        image, annotation = sample['image'], sample['annotation']
        # check image
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        if size is not None:
            self.assertEqual(image.shape, (3, size, size))
        # check annotation
        if not anno_is_none:
            self.assertIsInstance(annotation, Tensor)
            self.assertEqual(annotation.dtype, torch.int64)
            if size is not None:
                self.assertEqual(annotation.shape, (size, size))
        else:
            self.assertIsNone(annotation)

    def test_train_split(self):
        train_set = ADE20KSceneParse150(self.root, split='train')
        self.assertEqual(len(train_set), 20210)
        self.check_sample(train_set[0])

    def test_valid_split(self):
        valid_set = ADE20KSceneParse150(self.root, split='valid')
        self.assertEqual(len(valid_set), 2000)
        self.check_sample(valid_set[0])

    def test_test_split(self):
        test_set = ADE20KSceneParse150(self.root, split='test')
        self.assertEqual(len(test_set), 3352)
        self.check_sample(test_set[0], anno_is_none=True)

    def test_transform_fn(self):
        train_set = ADE20KSceneParse150(
            root=self.root,
            split='train',
            transform_fn=ADE20KSceneParse150Transform(size=640, flip_prob=0.5, mean=0.5, std=0.5),
        )
        self.check_sample(train_set[0], size=640)


class ADE20KSceneParse150Transform:
    def __init__(self, size, flip_prob, mean, std):
        self.size = size
        self.flip_prob = flip_prob
        self.mean = mean
        self.std = std

    def __call__(self, sample: dict):
        image, annotation = sample['image'], sample['annotation']
        # resize
        image = TF.resize(image, self.size, antialias=True)
        annotation = TF.resize(annotation.unsqueeze(0), self.size, interpolation=T.InterpolationMode.NEAREST).squeeze(0)
        # center crop
        image = TF.center_crop(image, self.size)
        annotation = TF.center_crop(annotation, self.size)
        # random horizontal flip
        if random.random() < self.flip_prob:
            image = TF.hflip(image)
            annotation = TF.hflip(annotation)
        # normalize
        image = TF.normalize(image, mean=self.mean, std=self.std)
        sample = {'image': image, 'annotation': annotation}
        return sample
