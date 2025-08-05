import unittest

import torch
import torchaudio
from torch import Tensor

from datasetx import SoundNet


class TestSoundNet(unittest.TestCase):

    root = '~/data/SoundNet'

    @classmethod
    def setUpClass(cls):
        print('\n\033[92m' + 'Testing SoundNet dataset...' + '\033[0m')

    def check_sample(self, sample: dict, sr: int = 22050, duration: float = None):
        # check sample, {image, audio, audio_sample_rate}
        self.assertIsInstance(sample, dict)
        self.assertEqual(sample.keys(), {'image', 'audio', 'audio_sample_rate'})
        image, audio, audio_sample_rate = sample['image'], sample['audio'], sample['audio_sample_rate']
        # check image
        self.assertIsInstance(image, Tensor)
        self.assertEqual(image.dtype, torch.float32)
        self.assertEqual(image.shape, (3, 256, 256))
        # check audio
        self.assertIsInstance(audio, Tensor)
        self.assertEqual(audio.dtype, torch.float32)
        if duration is not None:
            self.assertEqual(audio.shape, (int(duration * sr), ))
        # check audio sample rate
        self.assertIsInstance(audio_sample_rate, int)
        self.assertEqual(audio_sample_rate, sr)

    def test_train_split(self):
        train_set = SoundNet(self.root, split='train')
        self.assertEqual(len(train_set), 7086978)
        self.check_sample(train_set[0])

    def test_valid_split(self):
        valid_set = SoundNet(self.root, split='valid')
        self.assertEqual(len(valid_set), 518948)
        self.check_sample(valid_set[0])

    def test_transform_fn(self):
        train_set = SoundNet(
            root=self.root,
            split='train',
            transform_fn=SoundNetTransform(sr=16000, duration=10.24),
        )
        self.check_sample(train_set[0], sr=16000, duration=10.24)


class SoundNetTransform:
    def __init__(self, sr: int, duration: float):
        self.sr = sr
        self.duration = duration

    def __call__(self, sample: dict) -> dict:
        waveform, orig_sr = sample['audio'], sample['audio_sample_rate']
        waveform = waveform.unsqueeze(0)
        waveform, _ = self.random_segment(waveform, int(orig_sr * self.duration))
        waveform = self.resample(waveform, orig_sr, self.sr)
        waveform = self.pad(waveform, int(self.sr * self.duration))
        sample['audio'] = waveform.squeeze(0)
        sample['audio_sample_rate'] = self.sr
        return sample

    @staticmethod
    def random_segment(waveform: Tensor, target_length: int):
        waveform_length = waveform.shape[-1]
        if waveform_length - target_length <= 0:
            return waveform, 0
        random_start = int((waveform_length - target_length) * torch.rand(1).item())
        return waveform[:, random_start : random_start + target_length], random_start

    @staticmethod
    def resample(waveform: Tensor, source_sr: int, target_sr: int):
        waveform = torchaudio.functional.resample(waveform, source_sr, target_sr)
        return waveform

    @staticmethod
    def pad(waveform: Tensor, target_length: int):
        channels, waveform_length = waveform.shape
        if waveform_length == target_length:
            return waveform
        temp_wav = torch.zeros((channels, target_length), dtype=torch.float32)
        temp_wav[:, :waveform_length] = waveform
        return temp_wav
