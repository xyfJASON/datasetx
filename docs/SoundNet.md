# SoundNet

[Official website](https://soundnet.csail.mit.edu)

## Brief introduction

The authors downloaded over two million videos from Flickr by querying for popular tags and dictionary words, which resulted in over one year of continuous natural sound and video. The length of each video varies from a few seconds to several minutes.

## Statistics

**Numbers**: 7,605,926 images, 2,146,555 audios (each audio matches multiple images)

**Splits** (train / valid): 7,086,978 / 518,948 image-audio pairs

**Image Resolution**: 256×256

**Audio Length (duration × sampling rate)**: 1e3 ~ 4e7

**Audio Sampling Rate**: 22.05 kHz

## Files

|               Content               |           Filename           | Size  |
|:-----------------------------------:|:----------------------------:|:-----:|
|                MP3s                 |      mp3_public.tar.gz       | 359GB |
|           Image Features            | image_features_public.tar.gz | 88GB  |
|               Frames                |     frames_public.tar.gz     | 62GB  |
|       List of URLs of Videos        |       urls_public.txt        | 150MB |
| Lists of Videos and Train/Val Split |     lists_public.tar.gz      | 84MB  |

## Usage

### Download

```shell
wget http://data.csail.mit.edu/soundnet/mp3_public.tar.gz
wget http://data.csail.mit.edu/soundnet/image_features_public.tar.gz
wget http://data.csail.mit.edu/soundnet/frames_public.tar.gz
wget http://data.csail.mit.edu/soundnet/urls_public.txt
wget http://data.csail.mit.edu/soundnet/lists_public.tar.gz
```

### File structure

```text
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
```

### Example

```python
from datasetx import SoundNet

root = '~/data/SoundNet'  # path to downloaded dataset
train_set = SoundNet(root, split='train')
valid_set = SoundNet(root, split='valid')
print(len(train_set))  # 7086978
print(len(valid_set))  # 518948
print(train_set[0]['image'].shape)  # (3, 256, 256)
print(train_set[0]['audio'].shape)  # (1070831, )
```
