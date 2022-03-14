import os

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Subset

from data.confounder_utils import prepare_confounder_data, prepare_group_confounder_data
from data.label_shift_utils import prepare_label_shift_data

root_dir = './data/'

dataset_attributes = {
    'CelebA': {
        'root_dir': './celebA'
    },
    'CUB': {
        'root_dir': './cub'
    },
    "MetaDatasetCatDog":{
        'root_dir': "./MetaDatasetCatDog"
    },
    "CMNIST":{
        "root_dir": "./CMNIST"
    }
}

for dataset in dataset_attributes:
    dataset_attributes[dataset]['root_dir'] = os.path.join(root_dir, dataset_attributes[dataset]['root_dir'])

shift_types = ['confounder', 'label_shift_step']

def prepare_data(args, train, return_full_dataset=False):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]['root_dir']
    if args.shift_type=='confounder':
        return prepare_confounder_data(args, train, return_full_dataset)
    elif args.shift_type.startswith('label_shift'):
        assert not return_full_dataset
        return prepare_label_shift_data(args, train)

def prepare_cifar10_data(args):
    print('==> Preparing data..')
    if args.augment_data:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_data = datasets.CIFAR10(root="./cifar", train=True, download=True,
                                  transform=transform_train)
    train_data.n_groups = 1
    train_data.group_counts = torch.tensor([len(train_data)])
    train_data.group_str = lambda x: "whole"
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True, num_workers=0)

    test_data = datasets.CIFAR10(root="./cifar", train=False, download=True,
                                 transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100,
                                              shuffle=False, num_workers=0)
    test_data.n_groups = 1
    test_data.group_counts = torch.tensor([len(test_data)])
    test_data.group_str = lambda x: "whole"
    return train_data, test_data, train_loader, test_loader

def prepare_group_data(args, group_id, train_data=None, return_full_dataset=False):
    return prepare_group_confounder_data(args, group_id, train_data=train_data, return_full_dataset=return_full_dataset)

def log_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')

def log_meta_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        logger.write(f'    {data["train_data"].group_str(group_idx, train=True)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(f'    {data["val_data"].group_str(group_idx, train=True)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(f'    {data["test_data"].group_str(group_idx, train=False)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')

def log_amazon_data(data, logger):
    logger.write('Training Data...\n')
    for group_idx in range(data['train_data'].n_groups):
        if data["train_data"].group_counts()[group_idx] == 0: continue
        logger.write(
            f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n')
    logger.write('Validation Data...\n')
    for group_idx in range(data['val_data'].n_groups):
        logger.write(
            f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n')
    if data['test_data'] is not None:
        logger.write('Test Data...\n')
        for group_idx in range(data['test_data'].n_groups):
            logger.write(
                f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n')
