# DreamBooth

[GitHub](https://github.com/google/dreambooth) | [Hugging Face](https://huggingface.co/datasets/google/dreambooth)

## Brief introduction

> Copied from Hugging Face.

The dataset includes 30 subjects of 15 different classes. 9 out of these subjects are live subjects (dogs and cats) and 21 are objects. The dataset contains a variable number of images per subject (4-6). Images of the subjects are usually captured in different conditions, environments and under different angles.

## Statistics

**Numbers**: 750 (30 subjects × 25 prompts per subject)

**Resolution**: 615×615 ~ 2048×2048

## Usage

### Download

```shell
# download from GitHub
git clone https://github.com/google/dreambooth.git

# or download from Hugging Face
huggingface-cli download google/dreambooth --repo-type dataset --local-dir .
```

### File structure

```text
root
└── dataset
    ├── prompts_and_classes.txt
    ├── references_and_licenses.txt
    ├── backpack
    ├── backpack_dog
    ├── ...
    └── wolf_plushie
```

### Example

```python
from datasetx import DreamBooth

root = '~/data/dreambooth'   # path to downloaded dataset
dataset = DreamBooth(root=root)
print(len(dataset))  # 750
print(dataset[0]['image'].shape)  # (3, 1280, 1280)
print(dataset[0]['prompt'])  # a backpack in the jungle
```
