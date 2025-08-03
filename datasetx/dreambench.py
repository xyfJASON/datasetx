import os
from PIL import Image
from typing import Optional, Callable

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor


class DreamBench(Dataset):
    """The DreamBooth Benchmark Dataset.

    Please organize the dataset in the following file structure:

    root
    └── dataset
        ├── prompts_and_classes.txt
        ├── references_and_licenses.txt
        ├── backpack
        ├── ...
        └── wolf_plushie

    Reference:
      - https://github.com/google/dreambooth
      - https://huggingface.co/datasets/google/dreambooth
      - https://github.com/xichenpan/Kosmos-G/blob/main/scripts/remove_dreambench_multiimg.sh

    """

    name2info = {
        'backpack': ('backpack', '02'),
        'backpack_dog': ('backpack', '02'),
        'bear_plushie': ('stuffed animal', '03'),
        'berry_bowl': ('bowl', '02'),
        'can': ('can', '01'),
        'candle': ('candle', '02'),
        'cat': ('cat', '04'),
        'cat2': ('cat', '02'),
        'clock': ('clock', '03'),
        'colorful_sneaker': ('sneaker', '01'),
        'dog': ('dog', '02'),
        'dog2': ('dog', '02'),
        'dog3': ('dog', '05'),
        'dog5': ('dog', '00'),
        'dog6': ('dog', '02'),
        'dog7': ('dog', '01'),
        'dog8': ('dog', '04'),
        'duck_toy': ('toy', '01'),
        'fancy_boot': ('boot', '02'),
        'grey_sloth_plushie': ('stuffed animal', '04'),
        'monster_toy': ('toy', '04'),
        'pink_sunglasses': ('glasses', '04'),
        'poop_emoji': ('toy', '00'),
        'rc_car': ('toy', '03'),
        'red_cartoon': ('cartoon', '00'),
        'robot_toy': ('toy', '00'),
        'shiny_sneaker': ('sneaker', '01'),
        'teapot': ('teapot', '04'),
        'vase': ('vase', '02'),
        'wolf_plushie': ('stuffed animal', '04'),
    }

    object_prompts = [
        'a {} in the jungle',
        'a {} in the snow',
        'a {} on the beach',
        'a {} on a cobblestone street',
        'a {} on top of pink fabric',
        'a {} on top of a wooden floor',
        'a {} with a city in the background',
        'a {} with a mountain in the background',
        'a {} with a blue house in the background',
        'a {} on top of a purple rug in a forest',
        'a {} with a wheat field in the background',
        'a {} with a tree and autumn leaves in the background',
        'a {} with the Eiffel Tower in the background',
        'a {} floating on top of water',
        'a {} floating in an ocean of milk',
        'a {} on top of green grass with sunflowers around it',
        'a {} on top of a mirror',
        'a {} on top of the sidewalk in a crowded street',
        'a {} on top of a dirt road',
        'a {} on top of a white rug',
        'a red {}',
        'a purple {}',
        'a shiny {}',
        'a wet {}',
        'a cube shaped {}'
    ]

    live_subject_prompts = [
        'a {} in the jungle',
        'a {} in the snow',
        'a {} on the beach',
        'a {} on a cobblestone street',
        'a {} on top of pink fabric',
        'a {} on top of a wooden floor',
        'a {} with a city in the background',
        'a {} with a mountain in the background',
        'a {} with a blue house in the background',
        'a {} on top of a purple rug in a forest',
        'a {} wearing a red hat',
        'a {} wearing a santa hat',
        'a {} wearing a rainbow scarf',
        'a {} wearing a black top hat and a monocle',
        'a {} in a chef outfit',
        'a {} in a firefighter outfit',
        'a {} in a police outfit',
        'a {} wearing pink glasses',
        'a {} wearing a yellow shirt',
        'a {} in a purple wizard outfit',
        'a red {}',
        'a purple {}',
        'a shiny {}',
        'a wet {}',
        'a cube shaped {}'
    ]

    def __init__(
            self,
            root: str,
            transform_fn: Optional[Callable] = None,
    ):
        self.root = os.path.expanduser(root)
        self.transform_fn = transform_fn

        self.metadata = []
        for name, (cls, filename) in self.name2info.items():
            image_path = os.path.join(self.root, 'dataset', name, f'{filename}.jpg')
            prompts = self.object_prompts if (cls != 'cat' and cls != 'dog') else self.live_subject_prompts
            for prompt in prompts:
                self.metadata.append((image_path, prompt.format(cls)))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index: int):
        image_path, prompt = self.metadata[index]
        # read image
        image = Image.open(image_path).convert('RGB')
        image = to_tensor(image)
        sample = {'image': image, 'prompt': prompt}
        # apply transform
        if self.transform_fn is not None:
            sample = self.transform_fn(sample)
        return sample
