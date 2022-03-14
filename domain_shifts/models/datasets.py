import copy
import glob
import os
import pdb

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn import functional

class FMoW_Batched_Dataset(Dataset):
    """
    Batched dataset for FMoW. Allows for getting a batch of data given
    a specific domain index.
    """

    def __init__(self, args, dataset, split, batch_size, transform):
        self.split_array = dataset.split_array
        self.split_dict = dataset.split_dict
        split_mask = self.split_array == self.split_dict[split]
        split_idx = np.where(split_mask)[0]

        self.full_idxs = dataset.full_idxs[split_idx]
        self.chunk_size = dataset.chunk_size
        self.root = dataset.root

        self.metadata_array = dataset.metadata_array[split_idx]
        self.y_array = dataset.y_array[split_idx]

        self.eval = dataset.eval
        self.collate = dataset.collate
        self.metadata_fields = dataset.metadata_fields
        self.data_dir = dataset.data_dir
        self.transform = transform

        if args.group_by_label and split == 'train':
            self.domains = self.y_array
            self.domain_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
            # self.num_envs = len(self.domains.unique())
            self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
            print("reset domains with labels")
        else:
            domains = dataset.metadata_array[split_idx, :2]
            self.domain_indices = [torch.nonzero((domains == loc).sum(-1) == 2).squeeze(-1)
                                   for loc in domains.unique(dim=0)]
            # self.domains = self.metadata_array[:, :2]
            self.domains = torch.zeros(sum([len(idx) for idx in self.domain_indices]))
            for i, idxes in enumerate(self.domain_indices):
                self.domains[idxes] = i
            self.domains = self.domains.to(int)

            self.num_envs = len(self.domains.unique())
            self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
        self.domain_counts = [len(d) for d in self.domain_indices]
        print("domain counts: ", self.domain_counts)
        print("Domain number:", len(self.domain_counts))

        self.targets = self.y_array
        self.batch_size = batch_size

        unique_labels, counts = torch.unique(self.targets, return_counts=True)
        label2count = {int(label): count for label, count in zip(unique_labels, counts)}

        ratios = []
        ratios_2 = []
        ratios_3 = []
        larger_than_threshold_count = 0

        unique_classes = self.targets.unique()

        for loc in self.domains.unique():
            idxes = torch.where(self.domains == loc)
            # unique_class, numbers = torch.unique(self.targets[idxes], return_counts=True)
            # print(f"For domain [{loc}], Max: {torch.max(numbers)}, Min: {torch.min(numbers)}")
            # ratios.append(float(torch.max(numbers) / torch.min(numbers)))
            for c_idx, c in enumerate(unique_classes):
                k_yd = len(torch.where(self.targets[idxes] == c)[0])

                k_d = len(np.where(self.domains == loc)[0])
                k_y = label2count[int(c)]
                # if numbers[c_idx] / label2count[int(c)] > 0.02:
                # larger_than_threshold_count += 1
                # ratios.append(float(numbers[c_idx] / label2count[int(c)]))
                ratios.append(k_yd ** 2 / label2count[int(c)] / len(np.where(self.domains == loc)[0]))
                E_yd = k_y * k_d / len(self.targets) ** 2
                ratios_2.append((k_yd / len(self.targets) - E_yd) ** 2 / E_yd)
                E_yd = 1 / k_d / k_y
                ratios_3.append((k_yd / len(self.targets) - E_yd) ** 2 / E_yd)
                # ratios_3.append(abs(ratios[-1] - 1/(len(self.domains.unique()) * len(self.targets.unique()))) ** 2)
        print("whole length:", len(self.targets))
        print("domain number:", len(self.domains.unique()))
        print("class number:", len(self.targets.unique()))
        # print("larger than threshold 0.02:", larger_than_threshold_count)
        # print("Max:", np.max(ratios))
        print("Metric 1:", np.mean(ratios) * len(self.domains.unique()) * len(self.targets.unique()))
        print("Metric 2:", np.sum(ratios_2))
        print("Metric 3:", np.sum(ratios_3))
        print("Metric 4:", np.sqrt(np.sum(ratios_2) / min(len(unique_classes) - 1, len(self.domains.unique() - 1))))

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_random_batch(self, domain):
        """Return the next batch of the specified domain."""
        idx = np.random.choice(np.arange(len(self.batch_indices[domain])))
        batch_index = self.batch_indices[domain][idx]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_sample(self, domain, prev_idx, cross=False):
        domain = domain.item()
        if cross:
            domain = np.random.choice(np.setdiff1d(list(self.domain2idx.keys()), [domain]))
        d_idx = self.domain_indices[self.domain2idx[domain]]
        idx = np.random.choice(d_idx)
        while idx == prev_idx and len(d_idx) > 1:
            prev_idx = np.random.choice(d_idx)
        return self.tgransform(self.get_input(idx)), self.targets[idx], self.domains[idx]

    def get_input(self, idx):
        """Returns x for a given idx."""
        idx = self.full_idxs[idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx], idx

    def __len__(self):
        return len(self.targets)

class CivilComments_Batched_Dataset(Dataset):
    """
    Batched dataset for CivilComments. Allows for getting a batch of data given
    a specific domain index.
    """

    def __init__(self, args, train_data, batch_size=16):
        meta = torch.nonzero(train_data.metadata_array[:, :8] == 1)

        train_data._text_array = [train_data.dataset._text_array[i] for i in train_data.indices]
        self.dataset = train_data
        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._text_array

        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        self.transform = train_data.transform

        self.data = train_data._text_array
        self.targets = self.y_array

        if args.group_by_label:
            self.domains = self.y_array
            self.domain_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
            self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
            self.num_envs = len(np.unique(self.domains))
            print("reset domains with labels")
            print([len(d) for d in self.domain_indices])
        else:
            from wilds.common.grouper import CombinatorialGrouper
            grouper = CombinatorialGrouper(train_data.dataset, ['y', 'black'])

            group_array = grouper.metadata_to_group(train_data.dataset.metadata_array).numpy()
            group_array = group_array[np.where(
                train_data.dataset.split_array == train_data.dataset.split_dict[
                    'train'])]
            print("reset domains with labels and attribute black")

            self.domains = torch.tensor(group_array)
            self.domain_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
            self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
            self.num_envs = len(np.unique(self.domains))

            self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
            self.num_envs = len(self.domain_indices)

        self.batch_size = batch_size
        self.domain_counts = [len(d) for d in self.domain_indices]

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_random_batch(self, domain):
        """Return the next batch of the specified domain."""
        idx = np.random.choice(np.arange(len(self.batch_indices[domain])))
        batch_index = self.batch_indices[domain][idx]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_sample(self, domain, prev_idx, cross=False):
        domain = domain.item()
        if cross:
            domain = np.random.choice(np.setdiff1d(list(self.domain2idx.keys()), [domain]))
        d_idx = self.domain_indices[self.domain2idx[domain]]
        idx = np.random.choice(d_idx)
        while idx == prev_idx and len(d_idx) > 1:
            prev_idx = np.random.choice(d_idx)
        return self.transform(self.get_input(idx)), self.targets[idx], self.domains[idx]

    def get_input(self, idx):
        """Returns x for a given idx."""
        return self.data[idx]

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx], idx

    def __len__(self):
        return len(self.targets)

class GeneralWilds_Batched_Dataset(Dataset):
    """
    Batched dataset for Amazon, Camelyon and IwildCam. Allows for getting a batch of data given
    a specific domain index.
    """

    def __init__(self, args, train_data, batch_size=16, domain_idx=0):
        domains = train_data.metadata_array[:, domain_idx]

        train_data._input_array = [train_data.dataset._input_array[i] for i in train_data.indices]
        self.num_envs = len(domains.unique())

        self.metadata_array = train_data.metadata_array
        self.y_array = train_data.y_array
        self.data = train_data._input_array

        self.eval = train_data.eval
        self.collate = train_data.collate
        self.metadata_fields = train_data.metadata_fields
        self.data_dir = train_data.data_dir
        if 'iwildcam' in str(self.data_dir):
            self.data_dir = f'{self.data_dir}/train'
        self.transform = train_data.transform

        self.data = train_data._input_array
        self.targets = self.y_array
        if args.group_by_label:
            self.domains = self.y_array
            print("Reset domains with label array.")

        else:
            self.domains = domains

        if args.within_group:
            assert not args.group_by_label
            self.domains = self.domains * 2 + self.y_array

        self.num_envs = len(self.domains.unique())
        self.domain_indices = [torch.nonzero(self.domains == loc).squeeze(-1) for loc in self.domains.unique()]
        self.domain2idx = {loc.item(): i for i, loc in enumerate(self.domains.unique())}
        self.batch_size = batch_size
        self.domain_counts = [len(d) for d in self.domain_indices]


    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, d_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(d_idx[torch.randperm(len(d_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_random_batch(self, domain):
        """Return the next batch of the specified domain."""
        idx = np.random.choice(np.arange(len(self.batch_indices[domain])))
        batch_index = self.batch_indices[domain][idx]
        return torch.stack([self.transform(self.get_input(i)) for i in batch_index]), \
               self.targets[batch_index], self.domains[batch_index]

    def get_sample(self, domain, prev_idx, cross=False):
        domain = domain.item()
        if cross:
            domain = np.random.choice(np.setdiff1d(list(self.domain2idx.keys()), [domain]))
        d_idx = self.domain_indices[self.domain2idx[domain]]
        idx = np.random.choice(d_idx)
        while idx == prev_idx and len(d_idx) > 1:
            prev_idx = np.random.choice(d_idx)
        return self.transform(self.get_input(idx)), self.targets[idx], self.domains[idx]

    def get_input(self, idx):
        """Returns x for a given idx."""
        if isinstance(self.data_dir, str) and 'amazon' in self.data_dir:
            return self.data[idx]
        else:
            # All images are in the train folder
            img_path = f'{self.data_dir}/{self.data[idx]}'
            img = Image.open(img_path)
            if isinstance(self.data_dir, str) and not ('iwildcam' in self.data_dir):
                img = img.convert('RGB')
            return img

    def __getitem__(self, idx):
        return self.transform(self.get_input(idx)), \
               self.targets[idx], self.domains[idx], idx

    def __len__(self):
        return len(self.targets)

