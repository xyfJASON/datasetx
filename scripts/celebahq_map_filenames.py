"""Map the filenames of CelebA-HQ from (0, 29999) to the original CelebA dataset.

Before mapping:
    root
    ├── CelebA-HQ-img
    │   ├── 0.jpg
    │   ├── ...
    │   └── 29999.jpg
    └── CelebA-HQ-to-CelebA-mapping.txt

After mapping:
    root
    ├── CelebA-HQ-img
    │   ├── 000004.jpg
    │   ├── ...
    │   └── 202591.jpg
    ├── CelebA-HQ-img-backup
    │   ├── 0.jpg
    │   ├── ...
    │   └── 29999.jpg
    └── CelebA-HQ-to-CelebA-mapping.txt

"""

import os
import shutil
import argparse
import pandas as pd


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True, help='root directory of CelebA-HQ')
    args = parser.parse_args()
    args.root = os.path.expanduser(args.root)

    # Read the mapping file
    mapping = pd.read_table(os.path.join(args.root, 'CelebA-HQ-to-CelebA-mapping.txt'), sep=r'\s+', index_col=0)
    mapping_dict = {f'{i}.jpg': mapping.iloc[i]['orig_file'] for i in range(30000)}

    # Create a new directory `root/CelebA-HQ-img-tmp` to store the renamed images
    os.makedirs(os.path.join(args.root, 'CelebA-HQ-img-tmp'), exist_ok=True)
    for key, value in mapping_dict.items():
        if not os.path.isfile(os.path.join(args.root, 'CelebA-HQ-img', key)):
            # If file not found, remove the temporary directory and raise an error
            shutil.rmtree(os.path.join(args.root, 'CelebA-HQ-img-tmp'))
            raise ValueError(f"{os.path.join(args.root, 'CelebA-HQ-img', key)} does not exist")
        # Copy the file to the temporary directory
        shutil.copy(os.path.join(args.root, 'CelebA-HQ-img', key), os.path.join(args.root, 'CelebA-HQ-img-tmp', value))

    # Backup the original directory and rename the temporary directory
    shutil.move(os.path.join(args.root, 'CelebA-HQ-img'), os.path.join(args.root, 'CelebA-HQ-img-backup'))
    shutil.move(os.path.join(args.root, 'CelebA-HQ-img-tmp'), os.path.join(args.root, 'CelebA-HQ-img'))
    print('Done')
