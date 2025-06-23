# ADE20K-SceneParse150

[ADE20K official website](https://ade20k.csail.mit.edu/) | [SceneParse150 official website](http://sceneparsing.csail.mit.edu/) | [GitHub](https://github.com/CSAILVision/ADE20K) | [Papers with Code](https://paperswithcode.com/dataset/ade20k)

## Brief introduction

> Copied from paperswithcode.

The ADE20K semantic segmentation dataset contains more than 20K scene-centric images exhaustively annotated with pixel-level objects and object parts labels. There are totally 150 semantic categories, which include stuffs like sky, road, grass, and discrete objects like person, car, bed.

**Note**: According to this [issue](https://github.com/CSAILVision/ADE20K/issues/40#issuecomment-1611477294), the full ADE20K dataset has 3000+ class annotations, SceneParse150 (150 classes) is a subset of the annotations. But our users usually use ADE20K to refer to SceneParse150. So, if you want to compare with the leaderboard, use SceneParse150!

## Statistics

**Numbers**: 25,562

**Splits** (train / val / test): 20,210 / 2,000 / 3,352

**Resolution**: Inconsistent, mostly around 683×512

## Usage

### Download

```shell
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://data.csail.mit.edu/places/ADEchallenge/release_test.zip
```

### File structure

```text
root
├── ADEChallengeData2016
│   ├── annotations
│   │   ├── training
│   │   │   ├── ADE_train_00000001.png
│   │   │   ├── ...
│   │   │   └── ADE_train_00020210.png
│   │   └── validation
│   │       ├── ADE_val_00000001.png
│   │       ├── ...
│   │       └── ADE_val_00002000.png
│   ├── images
│   │   ├── training
│   │   │   ├── ADE_train_00000001.jpg
│   │   │   ├── ...
│   │   │   └── ADE_train_00020210.jpg
│   │   └── validation
│   │       ├── ADE_val_00000001.jpg
│   │       ├── ...
│   │       └── ADE_val_00002000.jpg
│   ├── objectInfo150.txt
│   └── sceneCategories.txt
└── release_test
    ├── list.txt
    ├── readme.txt
    └── testing
        ├── ADE_test_00000001.jpg
        ├── ...
        └── ADE_test_00003489.jpg
```

### Example

```python
from image_datasets import SceneParse150

root = '~/data/ADE20K/SceneParsing'   # path to downloaded dataset
train_set = SceneParse150(root=root, split='train')
valid_set = SceneParse150(root=root, split='valid')
test_set = SceneParse150(root=root, split='test')
print(len(train_set))  # 20210
print(len(valid_set))  # 2000
print(len(test_set))   # 3352
print(train_set[0])    # (<PIL.Image.Image image mode=RGB size=683x512 at 0x7F3EF3951F90>, <PIL.Image.Image image mode=L size=683x512 at 0x7F3EF2A06710>)
```
