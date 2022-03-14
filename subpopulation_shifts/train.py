import os

import numpy as np
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import (get_linear_schedule_with_warmup,
                          get_cosine_schedule_with_warmup)

from loss import LossComputer

device = torch.device("cuda")

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mix_up(args, x1, x2, y1, y2):

    # y1, y2 should be one-hot label, which means the shape of y1 and y2 should be [bsz, n_classes]

    length = min(len(x1), len(x2))
    x1 = x1[:length]
    x2 = x2[:length]
    y1 = y1[:length]
    y2 = y2[:length]

    n_classes = y1.shape[1]
    bsz = len(x1)
    l = np.random.beta(args.mix_alpha, args.mix_alpha, [bsz, 1])
    if len(x1.shape) == 4:
        l_x = np.tile(l[..., None, None], (1, *x1.shape[1:]))
    else:
        l_x = np.tile(l, (1, *x1.shape[1:]))
    l_y = np.tile(l, [1, n_classes])

    # mixed_input = l * x + (1 - l) * x2
    mixed_x = torch.tensor(l_x, dtype=torch.float32).to(x1.device) * x1 + torch.tensor(1-l_x, dtype=torch.float32).to(x2.device) * x2
    mixed_y = torch.tensor(l_y, dtype=torch.float32).to(y1.device) * y1 + torch.tensor(1-l_y, dtype=torch.float32).to(y2.device) * y2

    return mixed_x, mixed_y

def cut_mix_up(args, x1, x2, y1, y2):

    length = min(len(x1), len(x2))
    x1 = x1[:length]
    x2 = x2[:length]
    y1 = y1[:length]
    y2 = y2[:length]

    input = torch.cat([x1,x2])
    target = torch.cat([y1,y2])

    rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])

    lam = np.random.beta(args.alpha, args.alpha)
    target_a = target
    target_b = target[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

    return input, lam*target_a + (1-lam)*target_b

def mix_forward(args, group_len, x, y, g, y_onehot, model):

    if len(x) == 4:
        # LISA for CUB, CMNIST, CelebA
        if np.random.rand() < args.mix_ratio:
            mix_type = 1
        else:
            mix_type = 2

        if mix_type == 1:
            # mix different A within the same feature Y
            mix_group_1 = [x[0], x[1], y_onehot[0], y_onehot[1]]
            mix_group_2 = [x[2], x[3], y_onehot[2], y_onehot[3]]
        elif mix_type == 2:
            # mix different Y within the same feature A
            mix_group_1 = [x[0], x[2], y_onehot[0], y_onehot[2]]
            mix_group_2 = [x[1], x[3], y_onehot[1], y_onehot[3]]

        if args.cut_mix:
            mixed_x_1, mixed_y_1 = cut_mix_up(args, mix_group_1[0], mix_group_1[1], mix_group_1[2],
                                              mix_group_1[3])
            mixed_x_2, mixed_y_2 = cut_mix_up(args, mix_group_2[0], mix_group_2[1], mix_group_2[2],
                                              mix_group_2[3])
        else:
            mixed_x_1, mixed_y_1 = mix_up(args, mix_group_1[0], mix_group_1[1], mix_group_1[2],
                                          mix_group_1[3])
            mixed_x_2, mixed_y_2 = mix_up(args, mix_group_2[0], mix_group_2[1], mix_group_2[2],
                                          mix_group_2[3])

        all_mix_x = [mixed_x_1, mixed_x_2]
        all_mix_y = [mixed_y_1, mixed_y_2]
        all_group = torch.ones(
            len(mixed_x_1) + len(mixed_x_2)) * 3  # all the mixed samples are set to be from group 3
        all_y = torch.ones(len(mixed_x_1) + len(mixed_x_2)).cuda()
        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)

    else:
        # MetaDataset group by label, the mixup should be performed within the label group.
        all_mix_x, all_mix_y, all_group, all_y = [], [], [], []
        for i in range(group_len):
            bsz = len(x[i])

            if args.cut_mix:
                mixed_x, mixed_y = cut_mix_up(args, x[i][: bsz // 2], x[i][bsz // 2:], y_onehot[i][:bsz // 2],
                                              y_onehot[i][bsz // 2:])
                all_group.append(g[i][:len(mixed_x)])
                all_y.append(y[i][:len(mixed_x)])
                assert len(mixed_x) == len(all_y[-1])
            else:
                mixed_x, mixed_y = mix_up(args, x[i][:bsz // 2], x[i][bsz // 2:],
                                          y_onehot[i][:bsz // 2], y_onehot[i][bsz // 2:])
                all_group.append(g[i][:len(mixed_x)])
                all_y.append(y[i][:len(mixed_x)])

            all_mix_x.append(mixed_x)
            all_mix_y.append(mixed_y)

        all_mix_x = torch.cat(all_mix_x, dim=0)
        all_mix_y = torch.cat(all_mix_y, dim=0)
        all_group = torch.cat(all_group)
        all_y = torch.cat(all_y)

    outputs = model(all_mix_x.cuda())
    return outputs, all_y, all_group, all_mix_y

def run_epoch_mix_every_batch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
                         is_training, show_progress=False, log_every=50, scheduler=None, count=0):
    assert is_training
    model.train()
    if 'bert' in args.model:
        model.zero_grad()

    length = []
    data_iter = {}
    if "all" in loader: len_loader = len(loader) - 1
    else: len_loader = len(loader)

    if len_loader <= 10:
        for i in range(len_loader):
            length.append(len(loader[list(loader.keys())[i]]))
            data_iter[i] = iter(loader[list(loader.keys())[i]])
    else:
        for i in range(len_loader):
            length.append(len(loader[list(loader.keys())[i]]))

        group_loaders_idxes = np.random.permutation(len_loader)
        selected_data_iter_idxes = group_loaders_idxes[:4]
        count_group = 4
        for i, idx in enumerate(selected_data_iter_idxes):
            data_iter[i] = iter(loader[list(loader.keys())[idx]])

    # len_dataloader = np.sum(length) // 4
    len_dataloader = np.min(length)

    if show_progress:
        dataloader_iter = tqdm(range(len_dataloader))
    else:
        dataloader_iter = range(len_dataloader)

    for batch_idx, it in enumerate(dataloader_iter):
        x, y, g, y_onehot = [], [], [], []

        # selected_data_iter_idxes = np.random.choice(np.arange(len_loader), 4, replace=False)
        if len_loader <= 4:
            selected_data_iter_idxes = np.arange(len_loader)
        else:
            selected_data_iter_idxes = np.random.choice(np.arange(len_loader), 4, replace=False)

        for i in selected_data_iter_idxes:
            try:
                tmp_x, tmp_y, tmp_g, tmp_y_onehot, _ = data_iter[i].next()
            except:
                data_iter[i] = iter(loader[list(loader.keys())[i]])
                tmp_x, tmp_y, tmp_g, tmp_y_onehot, _ = data_iter[i].next()

            x.append(tmp_x)
            y.append(tmp_y)
            g.append(torch.ones(len(tmp_y)) * i)
            y_onehot.append(tmp_y_onehot)

        outputs, all_y, all_group, all_mix_y = mix_forward(args=args,
                                                           group_len=len(data_iter),
                                                           x=x, y=y, g=g, y_onehot=y_onehot, model=model)

        loss_main = loss_computer.loss(outputs, all_y.cuda(), all_group.cuda(), is_training,
                                       mix_up=args.lisa_mix_up, y_onehot=all_mix_y.cuda())

        if 'bert' in args.model:
            loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        else:
            optimizer.zero_grad()
            loss_main.backward()
            optimizer.step()

        if (count+1) % log_every == 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            loss_computer.reset_stats()
        count+=1
    return count

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):

    if is_training:
        model.train()
        if 'bert' in args.model:
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):

        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda() for t in batch)

            x = batch[0]
            y = batch[1]
            g = batch[2]
            y_onehot = None

            outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training, mix_up=args.lisa_mix_up, y_onehot=y_onehot)

            if is_training:
                if 'bert' in args.model:
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()

def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, n_classes, epoch_offset, csv_name=None, exp_string=None):
    model = model.cuda()

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        args,
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight)

    # BERT uses its own scheduler and optimizer
    if 'bert' in args.model:
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.lr,
            eps=args.adam_epsilon)

        if args.lisa_mix_up:
            lengths = []
            for key in dataset['train_loader'].keys():
                lengths.append(len(dataset['train_loader'][key]))
            # If there are 5 groups, then we will choose the length of the
            # second to least long loader as the epoch length
            length = np.sum(lengths) // 4
        else:
            length = len(dataset['train_loader'])
        
        t_total = length * args.n_epochs

        print(f'\nt_total is {t_total}\n')
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=args.warmup_steps,
            t_total=t_total)

    else:
        if args.lisa_mix_up:
            lengths = []
            for key in dataset['train_loader'].keys():
                lengths.append(len(dataset['train_loader'][key]))
            length = np.sum(lengths) // 4
        else:
            length = len(dataset['train_loader'])

        t_total = length * args.n_epochs

        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr,
                weight_decay=args.weight_decay)
        else:
            raise ValueError(f"{args.optimizer} not recognized")

        if args.scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                'min',
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08)

        elif args.scheduler == 'linear_schedule_with_warmup':
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_training_steps=t_total,
                num_warmup_steps=args.num_warmup_steps)

            step_every_batch = True
            use_metric = False

        elif args.scheduler == 'cosine_schedule_with_warmup':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_training_steps=t_total,
                num_warmup_steps=args.num_warmup_steps)

        elif args.scheduler == 'StepLR':
            scheduler = StepLR(optimizer,
                               t_total,
                               gamma=args.step_gamma)

        else:
            scheduler = None

    best_val_acc = 0
    count = 1
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        if args.lisa_mix_up:
            count = run_epoch_mix_every_batch(epoch, model, optimizer,
                             dataset['train_loader'],
                             train_loss_computer,
                             logger, train_csv_logger, args,
                             is_training=True,
                             show_progress=args.show_progress,
                             log_every=args.log_every,
                             scheduler=scheduler,
                             count=count
                             )

        else:
            run_epoch(
                epoch, model, optimizer,
                dataset['train_loader'],
                train_loss_computer,
                logger, train_csv_logger, args,
                is_training=True,
                show_progress=args.show_progress,
                log_every=args.log_every,
                scheduler=scheduler)

        logger.write(f'\nEpoch {epoch}, Validation:\n')
        val_loss_computer = LossComputer(
            args,
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha,
            is_val=True)
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        if dataset['test_data'] is not None:
            logger.write(f'\nEpoch {epoch}, Testing:\n')
            test_loss_computer = LossComputer(
                args,
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha)
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                logger, test_csv_logger, args,
                is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            # if args.robust or args.reweight_groups:
            #     curr_val_acc = min(val_loss_computer.avg_group_acc)
            # else:
            #     curr_val_acc = val_loss_computer.avg_acc

            curr_val_acc = val_loss_computer.worst_group_acc

            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')
