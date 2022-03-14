import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from data import folds
from data.data import dataset_attributes, shift_types, prepare_data, log_data, log_meta_data
from data.dro_dataset import DRODataset
from data.folds import Subset
from models import model_attributes
from train import train
from utils import ParseKwargs
from utils import set_seed, Logger, CSVBatchLogger, log_args, construct_loader, Identity


def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), default="CMNIST")
    parser.add_argument('-s', '--shift_type', choices=shift_types, default='confounder')
    # Confounders
    parser.add_argument('-t', '--target_name', default='waterbird_complete95')
    parser.add_argument('-c', '--confounder_names', nargs='+', default=['forest2water2'])
    # Resume?
    parser.add_argument('--resume', default=False, action='store_true')
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--reweight_groups', action='store_true', default=False)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--val_fraction', type=float, default=0.1)
    parser.add_argument("--dog_group", type=int, default=4)
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--robust_step_size', default=0.01, type=float)
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')


    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default='SGD')
    parser.add_argument('--scheduler', type=str, default=None)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)
    parser.add_argument('--lisa_mix_up', action='store_true', default=False)
    parser.add_argument("--mix_ratio", default=0.5, type=float)
    parser.add_argument("--mix_alpha", default=2, type=float)
    parser.add_argument("--cut_mix", default=False, action='store_true')
    parser.add_argument("--num_warmup_steps", default=0, type=int)

    # Misc
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--save_best', action='store_true', default=False)
    parser.add_argument('--save_last', action='store_true', default=False)
    parser.add_argument("--fold", default=None)
    parser.add_argument("--num_folds_per_sweep", type=int, default=5)
    parser.add_argument("--num_sweeps", type=int, default=4)
    parser.add_argument("--is_featurizer", type=int, default=True)
    parser.add_argument("--step_gamma", type=float, default=0.96)
    parser.add_argument("--group_by_label", action='store_true', default=False)

    args = parser.parse_args()
    check_args(args)

    exp_string = args.dataset
    if args.robust: exp_string += '_robust'
    if args.reweight_groups: exp_string += '_reweight'
    if eval(args.generalization_adjustment) > 0: exp_string += '_adjust'
    if args.lisa_mix_up:
        exp_string += f'_mix_up_{args.mix_alpha}'
        exp_string += '_cut_mix' if args.cut_mix else ""
    if args.dataset == 'MetaDatasetCatDog':
        exp_string += f'_dog_{int(args.dog_group)}'
    if args.weight_decay >= 0.01:
        exp_string += f"_penalty_{args.weight_decay}"
    exp_string += f'_{args.seed}'

    # BERT-specific configs copied over from run_glue.py
    if 'bert' in args.model:
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = Logger(os.path.join(args.log_dir, f'{exp_string}_log.txt'), mode)
    # Record args
    log_args(args, logger)

    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    id_val_data, id_test_data = None, None

    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)
    else:
        raise NotImplementedError

    train_data.n_groups = len(np.unique(train_data.get_group_array()))
    val_data.n_groups = len(np.unique(val_data.get_group_array()))
    test_data.n_groups = len(np.unique(test_data.get_group_array()))

    if args.fold:
        train_data, val_data = folds.get_fold(
            train_data,
            args.fold,
            cross_validation_ratio=(1 / args.num_folds_per_sweep),
            num_valid_per_point=args.num_sweeps,
            seed=args.seed,
        )

    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':4, 'pin_memory':False}

    val_loader = construct_loader(val_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs)
    id_val_loader = construct_loader(id_val_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs)
    test_loader = construct_loader(test_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs)
    id_test_loader = construct_loader(id_test_data, train=False, reweight_groups=None, loader_kwargs=loader_kwargs)

    data = {}
    if args.lisa_mix_up:
        train_loader = {}
        for i in range(train_data.n_groups):
            idxes = np.where(train_data.get_group_array() == i)[0]
            if len(idxes) == 0: continue
            temp_train_data = DRODataset(Subset(train_data, idxes), process_item_fn=None, n_groups=train_data.n_groups,
                       n_classes=train_data.n_classes, group_str_fn=train_data.group_str)

            train_loader[i] = temp_train_data.get_loader(train=True, reweight_groups=False, **loader_kwargs)

    else:
        print("Get loader")

        print("length of train_data:", len(train_data))

        train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)

    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['id_val_loader'] = id_val_loader
    data['id_test_loader'] = id_test_loader

    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    data['id_val_data'] = id_val_data
    data['id_test_data'] = id_test_data

    n_classes = train_data.n_classes

    if "Meta" in args.dataset:
        log_meta_data(data, logger)
    else:
        log_data(data, logger)


    ## Initialize model
    pretrained = not args.train_from_scratch
    if resume:
        model = torch.load(os.path.join(args.log_dir, 'last_model.pth'))
        d = train_data.input_size()[0]
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'densenet121':
        model = torchvision.models.densenet121(pretrained=pretrained)
        d = model.classifier.in_features
        model.classifier = nn.Linear(d, n_classes)

    elif 'bert' in args.model:
        if args.is_featurizer:
            if args.model == 'bert':
                from bert.bert import BertFeaturizer
                featurizer = BertFeaturizer.from_pretrained("bert-base-uncased", **args.model_kwargs)
                classifier = nn.Linear(featurizer.d_out, 5 if args.dataset == "Amazon" else n_classes)
                model = torch.nn.Sequential(featurizer, classifier)
            elif args.model == 'distilbert':
                from bert.distilbert import DistilBertFeaturizer
                featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased", **args.model_kwargs)
                classifier = nn.Linear(featurizer.d_out, 5 if args.dataset == "Amazon" else n_classes)
                model = torch.nn.Sequential(featurizer, classifier)
            else:
                raise NotImplementedError

        else:
            from bert.bert import BertClassifier
            model = BertClassifier.from_pretrained(
                'bert-base-uncased',
                num_labels=512,
                **args.model_kwargs)


    else:
        raise ValueError('Model not recognized.')

    logger.flush()

    ## Define the objective
    if args.hinge:
        assert args.dataset in ['CelebA', 'CUB'] # Only supports binary
        def hinge_loss(yhat, y):
            # The torch loss takes in three arguments so we need to split yhat
            # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
            # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
            # so we need to swap yhat[:, 0] and yhat[:, 1]...
            torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
            y = (y.float() * 2.0) - 1.0
            return torch_loss(yhat[:, 1], yhat[:, 0], y)
        criterion = hinge_loss
    else:
        criterion = torch.nn.CrossEntropyLoss(reduction='none')

    if resume:
        df = pd.read_csv(os.path.join(args.log_dir, f'{exp_string}_test.csv'))
        epoch_offset = df.loc[len(df)-1,'epoch']+1
        logger.write(f'starting from epoch {epoch_offset}')
    else:
        epoch_offset=0

    train_csv_logger = CSVBatchLogger(args, os.path.join(args.log_dir, f'{exp_string}_train.csv'), train_data.n_groups, mode=mode)
    val_csv_logger =  CSVBatchLogger(args, os.path.join(args.log_dir, f'{exp_string}_val.csv'), val_data.n_groups, mode=mode)
    test_csv_logger =  CSVBatchLogger(args, os.path.join(args.log_dir, f'{exp_string}_test.csv'), test_data.n_groups, mode=mode)

    train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args,
          n_classes, csv_name=args.fold, exp_string=exp_string, epoch_offset=epoch_offset)

    train_csv_logger.close()
    val_csv_logger.close()
    test_csv_logger.close()

    # print results
    val_csv = pd.read_csv(os.path.join(args.log_dir, f'{exp_string}_val.csv'))
    test_csv = pd.read_csv(os.path.join(args.log_dir, f'{exp_string}_test.csv'))
    idx = np.argmax(val_csv['worst_group_acc'].values)
    print(test_csv[['worst_group_acc', 'mean_differences', "group_avg_acc", "avg_acc"]].iloc[idx])

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio



if __name__=='__main__':
    main()
