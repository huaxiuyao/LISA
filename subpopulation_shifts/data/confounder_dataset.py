import os
import pdb

import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset
from data.folds import Subset


class ConfounderDataset(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None, augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.group_array)

    def __getitem__(self, idx):
        g = self.group_array[idx]
        y = self.y_array[idx]

        # if model_attributes[self.model_type]['feature_type']=='precomputed':
        if self.precomputed:
            x = self.features_mat[idx]
            if not self.pretransformed:
                if self.split_array[idx] == 0:
                    x = self.train_transform(x)
                else:
                    x = self.eval_transform(x)

            assert not isinstance(x, list)
        else:
            if not self.mix_array[idx]:
                x = self.get_image(idx)
            else:
                idx_1, idx_2 = self.mix_idx_array[idx]
                x1, x2 = self.get_image(idx_1), self.get_image(idx_2)

                l = self.mix_weight_array[idx]

                x = l * x1 + (1-l) * x2

        if self.mix_up:
            y_onehot = self.y_array_onehot[idx]

            try:
                true_g = self.domains[idx]
            except:
                true_g = None

            if true_g is None:
                return x, y, g, y_onehot, idx

            else:
                return x, y, true_g, y_onehot, idx

        else:
            return x, y, g, idx

    def refine_dataset(self):
        for name, split_id in self.split_dict.items():
            idxes = np.where(self.split_array == split_id)
            group_counts = (torch.arange(self.n_groups).unsqueeze(1)==torch.tensor(self.group_array[idxes])).sum(1).float()
            unique_group_id = torch.where(group_counts > 0)[0]
            # unique_group_id, counts = np.unique(self.group_array[idxes], return_counts=True)
            # unique_group_id = unique_group_id[np.where(counts) > 0]
            group_dict = {id: new_id for new_id, id in enumerate(unique_group_id.tolist())}
            self.group_array[idxes] = np.array([group_dict[id] for id in self.group_array[idxes]])

    def get_image(self, idx):
        # img_filename = os.path.join(
        #     self.data_dir,
        #     self.filename_array[idx])
        img_filename = self.filename_array[idx]
        img = Image.open(img_filename)
        if self.RGB:
            img = img.convert("RGB")
        # Figure out split and transform accordingly
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
            img = self.eval_transform(img)
        # Flatten if needed
        if model_attributes[self.model_type]['flatten']:
            assert img.dim() == 3
            img = img.view(-1)
        return img


    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            # assert split in ('train','val','test'), split+' is not a valid split'
            mask = self.split_array == self.split_dict[split]
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if train_frac<1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
