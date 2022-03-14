import glob
from enum import unique
import os
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset
from data.confounder_dataset import ConfounderDataset


class MetaDatasetCatDog(ConfounderDataset):

    def __init__(self, args, root_dir,
                 target_name, confounder_names,
                 augment_data=False,
                 model_type=None,
                 mix_up=False,
                 mix_alpha=2,
                 mix_unit='group',
                 mix_type=1,
                 mix_freq='batch',
                 mix_extent=None,
                 group_id=None,
                 dataset=None):
        self.args = args
        self.mix_up = mix_up
        self.mix_alpha = mix_alpha
        self.mix_unit = mix_unit
        self.mix_type = mix_type
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.n_classes = 2
        self.n_groups = 4

        self.train_data_dir = os.path.join(self.root_dir, "train")
        self.test_data_dir = os.path.join(self.root_dir, 'test')

        self.n_confounders = 1
        self.RGB = True

        cat_dict = {
            0: ["sofa"],
            1: ["bed"]
        }

        test_groups = {
            "cat": ["shelf"],
            "dog": ["shelf"]
        }

        self.test_groups = test_groups

        if args.dog_group == 1:
            dog_dict = {
                0: ['cabinet'],
                1: ['bed']
            }
        elif args.dog_group == 2:
            dog_dict = {
                0: ['bag'],
                1: ['box']
            }
        elif args.dog_group == 3:
            dog_dict = {
                0: ['bench'],
                1: ['bike']
            }
        elif args.dog_group == 4:
            dog_dict = {
                0: ['boat'],
                1: ['surfboard']
            }
        else:
            raise NotImplementedError

        train_groups = {
            "cat": cat_dict,
            "dog": dog_dict
        }
        self.train_groups = train_groups

        self.train_filename_array, self.train_group_array, self.train_y_array = self.get_data(train_groups,
                                                                                              is_training=True)
        self.test_filename_array, self.test_group_array, self.test_y_array = self.get_data(test_groups,
                                                                                           is_training=False)

        train_idxes = np.arange(len(self.train_group_array))
        ###############################
        # split train and validation set
        np.random.seed(100)
        val_idxes = np.random.choice(train_idxes, size=int(0.1 * len(train_idxes)), replace=False)
        train_idxes = np.setdiff1d(train_idxes, val_idxes)
        ###############################

        # define the split array
        self.train_split_array = np.zeros(len(self.train_group_array))
        self.train_split_array[val_idxes] = 1
        self.test_split_array = 2 * np.ones(len(self.test_group_array))

        self.filename_array = np.concatenate([self.train_filename_array, self.test_filename_array])
        self.group_array = np.concatenate([self.train_group_array, self.test_group_array])
        self.split_array = np.concatenate([self.train_split_array, self.test_split_array])
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.y_array = np.concatenate([self.train_y_array, self.test_y_array])
        self.y_array_onehot = torch.zeros(len(self.y_array), self.n_classes)
        self.y_array_onehot = self.y_array_onehot.scatter_(1, torch.tensor(self.y_array).unsqueeze(1), 1).numpy()
        self.mix_array = [False] * len(self.y_array)

        if group_id is not None:
            idxes = np.where(self.group_array == group_id)
            self.filename_array = self.filename_array[idxes]
            self.group_array = self.group_array[idxes]
            self.split_array = self.split_array[idxes]
            self.y_array = self.y_array[idxes]
            self.y_array_onehot = self.y_array_onehot[idxes]

        self.precomputed = False

        self.train_transform = get_transform_cub(
            self.model_type,
            train=True,
            augment_data=augment_data)
        self.eval_transform = get_transform_cub(
            self.model_type,
            train=False,
            augment_data=augment_data)

        self.domains = self.group_array
        if args.group_by_label:
            print("reset groups by labels")
            self.group_array = self.y_array
            self.n_groups = 2
        else:
            self.n_groups = len(np.unique(self.group_array))


    def resplit_data(self, train_filename_array, train_group_array, train_y_array,
                     test_filename_array, test_group_array, test_y_array, ratio):
        # Change the group id 1 in train_group_array to 2
        train_group_array[np.where(train_group_array == 1)] = 2

        # Change the group_id 0, 1 in test_group_array to 1 and 3
        test_group_array[np.where(test_group_array == 1)] = 3
        test_group_array[np.where(test_group_array == 0)] = 1

        # Concat train and test datasets
        all_filename_array = np.concatenate([train_filename_array, test_filename_array])
        all_group_array = np.concatenate([train_group_array, test_group_array])
        all_y_array = np.concatenate([train_y_array, test_y_array])

        # 0: cat indoor
        # 1: cat outdoor
        # 2: dog outdoor
        # 3: dog indoor

        # Check length
        lengths = [1127, 446, 1109, 899]
        ratios = [ratio, 1 - ratio, ratio, 1 - ratio]

        # For example, if ratio == 0.8, then it means we choose:
        # 80% cat indoor, 20% cat outdoor,80% dog outdoor, 20% dog indoor

        # resplit dataset
        train_indexes = []
        for group_id in range(4):
            indexes = np.where(all_group_array == group_id)[0]
            assert len(indexes) == lengths[group_id]
            np.random.seed(100)
            train_indexes.append(np.random.choice(indexes, size=int(ratios[group_id] * len(indexes)), replace=False))

        train_indexes = np.concatenate(train_indexes)
        test_indexes = np.setdiff1d(np.arange(len(all_group_array)), train_indexes)

        return all_filename_array[train_indexes], all_group_array[train_indexes], all_y_array[train_indexes], \
               all_filename_array[test_indexes], all_group_array[test_indexes], all_y_array[test_indexes]

    def get_group_array(self):
        return self.group_array

    def get_label_array(self):
        return self.y_array

    def group_str(self, group_idx, train=False):
        if not train:
            if group_idx < len(self.test_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.test_groups['cat'][group_idx]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.test_groups['dog'][group_idx - len(self.test_groups['cat'])]}"

        else:
            if group_idx < len(self.train_groups['cat']):
                group_name = f'y = cat'
                group_name += f", attr = {self.train_groups['cat'][group_idx][0]}"
            else:
                group_name = f"y = dog"
                group_name += f", attr = {self.train_groups['dog'][group_idx - len(self.train_groups['cat'])][0]}"

        # bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        # for attr_idx, attr_name in enumerate(self.confounder_names):
        #     group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name

    def get_data(self, groups, is_training):
        filenames = []
        group_ids = []
        ys = []
        id_count = 0
        animal_count = 0
        for animal in groups.keys():
            if is_training:
                for _, group_animal_data in groups[animal].items():
                    for group in group_animal_data:
                        for file in os.listdir(f"{self.train_data_dir}/{animal}/{animal}({group})"):
                            filenames.append(os.path.join(f"{self.train_data_dir}/{animal}/{animal}({group})", file))
                            group_ids.append(id_count)
                            ys.append(animal_count)
                    id_count += 1

            else:
                for group in groups[animal]:
                    for file in os.listdir(f"{self.test_data_dir}/{animal}/{animal}({group})"):
                        filenames.append(os.path.join(f"{self.test_data_dir}/{animal}/{animal}({group})", file))
                        group_ids.append(id_count)
                        ys.append(animal_count)
                    id_count += 1

            animal_count += 1

        return filenames, np.array(group_ids), np.array(ys)


def get_transform_cub(model_type, train, augment_data):
    # Borrowed from cub_dataset.py

    scale = 256.0 / 224.0
    target_resolution = model_attributes[model_type]['target_resolution']
    assert target_resolution is not None

    if (not train) or (not augment_data):
        # Resizes the image to a slightly larger square then crops the center.
        transform = transforms.Compose([
            transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
