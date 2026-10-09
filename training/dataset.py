"""CSV word dataset with five training-only references for a single writer."""
import csv
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

def tensor(path):
    with Image.open(path) as im:
        assert im.size == (256, 64), f'Unprepared image: {path}'
        arr = np.array(im.convert('RGB'), dtype=np.float32)
    return torch.from_numpy(arr.transpose(2,0,1) / 127.5 - 1.)

class WordDataset(Dataset):
    def __init__(self, root, split, vocabulary, seed=42):
        self.root, self.split, self.seed = Path(root), split, seed
        self.rows = list(csv.DictReader((self.root/split/'labels.csv').open(encoding='utf-8',newline='')))
        train = list(csv.DictReader((self.root/'train/labels.csv').open(encoding='utf-8',newline='')))
        self.references = [r for r in train if len(r['text'])>=4]
        if len(self.references)<6:
            raise ValueError('Need at least six reviewed training words >=4 characters to exclude the target from five references')
        if not self.rows:
            raise ValueError(f'Empty {split} dataset')
        for row in self.rows:
            if set(row['text'])-set(vocabulary) or len(row['text'].split()) != 1:
                raise ValueError(f'Unsupported or multiword target: {row}')
            if row['type'] != 'word':
                raise ValueError('Prepare and visually review word-level data first')
            if not (self.root/split/'images'/row['filename']).is_file():
                raise FileNotFoundError(row['filename'])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        candidates = [r for r in self.references if not (self.split=='train' and r['filename']==row['filename'])]
        refs = random.Random(self.seed+index).sample(candidates,5)
        return {'image': tensor(self.root/self.split/'images'/row['filename']),
                'text': row['text'], 'filename': row['filename'],
                'style_images': torch.stack([tensor(self.root/'train/images'/r['filename']) for r in refs]),
                'reference_filenames': [r['filename'] for r in refs]}
