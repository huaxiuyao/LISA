import pdb

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

class DRODataset(Dataset):
    def __init__(self, dataset, process_item_fn, n_groups, n_classes, group_str_fn):
        self.dataset = dataset
        self.process_item = process_item_fn
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.group_str = group_str_fn
        # group_array = []
        # y_array = []
        # for batch in self:
        #     group_array.append(batch[2])
        #     y_array.append(batch[1])
        # self._group_array = torch.LongTensor(group_array)
        # self._y_array = torch.LongTensor(y_array)


        # self._group_array = torch.LongTensor(dataset.dataset.group_array[dataset.indices])
        # self._y_array = torch.LongTensor(dataset.dataset.y_array[dataset.indices])

        self._group_array = torch.LongTensor(dataset.get_group_array())
        self._y_array = torch.Tensor(dataset.get_label_array())
        self._group_counts = (torch.arange(self.n_groups).unsqueeze(1)==self._group_array).sum(1).float()
        self._group_counts = self._group_counts[np.where(self._group_counts > 0)]
        self._y_counts = (torch.arange(self.n_classes).unsqueeze(1)==self._y_array).sum(1).float()
        self.group_indices = {loc.item():torch.nonzero(self._group_array == loc).squeeze(-1)
                               for loc in self._group_array.unique()}
        self.distinct_groups = np.unique(self._group_array)

        assert len(self._group_array) == len(self.dataset)

    def get_sample(self, g, idx, cross=False):
        g = g.item()
        if cross:
            g = np.random.choice(np.setdiff1d(self.distinct_groups, [g]))
        new_idx = np.random.choice(self.group_indices[g].numpy())
        # while new_idx == idx:
        #     new_idx = np.random.choice(self.group_indices[g].numpy())
        # if idx >= len(self.dataset):
        #     pdb.set_trace()
        return self.dataset[new_idx]



    def __getitem__(self, idx):

        if self.process_item is None:
            return self.dataset[idx]
        else:
            return self.process_item(self.dataset[idx])

    def __len__(self):
        return len(self.dataset)

    def group_counts(self):
        return self._group_counts

    def class_counts(self):
        return self._y_counts

    def input_size(self):
        for sample in self:
            x = sample[0]
            return x.size()

    def get_group_array(self):
        if self.process_item is None:
            return self.dataset.get_group_array()
        else:
            raise NotImplementedError

    def get_label_array(self):
        if self.process_item is None:
            return self.dataset.get_label_array()
        else:
            raise NotImplementedError

    def get_loader(self, train, reweight_groups, **kwargs):
        if not train: # Validation or testing
            assert reweight_groups is None
            shuffle = False
            sampler = None
        elif not reweight_groups: # Training but not reweighting
            shuffle = True
            sampler = None
        else: # Training and reweighting
            # When the --robust flag is not set, reweighting changes the loss function
            # from the normal ERM (average loss over each training example)
            # to a reweighted ERM (weighted average where each (y,c) group has equal weight) .
            # When the --robust flag is set, reweighting does not change the loss function
            # since the minibatch is only used for mean gradient estimation for each group separately
            group_weights = len(self)/self._group_counts
            weights = group_weights[self._group_array]

            assert not np.isnan(weights).any()

            # Replacement needs to be set to True, otherwise we'll run out of minority samples
            sampler = WeightedRandomSampler(weights, len(self), replacement=True)
            shuffle = False

        loader = DataLoader(
            self,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs)
        return loader
