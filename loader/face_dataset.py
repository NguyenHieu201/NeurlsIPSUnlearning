import os

import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# Kaggle NeurIPS Unlearning Challenge example notebook
# Helper functions for loading the hidden dataset.

def load_example(df_row):
    resize_ts = torchvision.transforms.Resize((32, 32), antialias=True)
    image = torchvision.io.read_image(df_row['path'])
    image = resize_ts(image)
    result = {
        'image': image,
        'age': df_row['age'],
    }
    return result


class HiddenDataset(Dataset):
    '''The hidden dataset.'''

    def __init__(self, split='train'):
        super().__init__()
        self.examples = []

        df = pd.read_csv(
            f'{split}.csv')
        df = df.sort_values(by='path')
        df.apply(lambda row: self.examples.append(load_example(row)), axis=1)
        if len(self.examples) == 0:
            raise ValueError('No examples.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image = example['image']
        image = image.to(torch.float32)
        example['image'] = image
        return example


def get_dataset(batch_size, data_dir, splits=['retain', 'forget', 'validation']):
    '''Get the dataset.'''
    retain_ds = HiddenDataset(split=os.path.join(data_dir, splits[0]))
    forget_ds = HiddenDataset(split=os.path.join(data_dir, splits[1]))
    val_ds = HiddenDataset(split=os.path.join(data_dir, splits[2]))

    retain_loader = DataLoader(retain_ds, batch_size=batch_size, shuffle=True)
    forget_loader = DataLoader(forget_ds, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return retain_loader, forget_loader, validation_loader
