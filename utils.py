import os
import random
import time

import numpy as np
import torch
import tqdm

from dataset import *


def seed_everything(seed=3407):
    '''set seed for deterministic training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cuda(x):
    return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


def get_time_suffix():
    ''' get current time, return string in format: year_month_day, e.g., 2021_11_11 '''
    time_struct = time.localtime(time.time())
    year, month, day = time_struct.tm_year, time_struct.tm_mon, time_struct.tm_mday
    time_suffix = str(year) + '_' + str(month) + '_' + str(day)
    return time_suffix


# coefficients of exponentially increase beta with epoch
# y=a*(0.5^(0.01*(x-1)))+c, each key-value represent "epoch_ratio: [a, c]"
increase_coe_dict = {0.10: [-0.099211, 0.099211],
                     0.20: [-0.048625, 0.048625],
                     0.30: [-0.032949, 0.032949],
                     0.35: [-0.028577, 0.028577],
                     0.40: [-0.025330, 0.025330],
                     0.45: [-0.032949, 0.032949],
                     0.50: [-0.020835, 0.020835],
                     0.60: [-0.017875, 0.017875],
                     0.70: [-0.015783, 0.015783],
                     0.80: [-0.014230, 0.014230],
                     0.85: [-0.013594, 0.013594],
                     0.90: [-0.013033, 0.013033],
                     0.95: [-0.012532, 0.012532],
                     1.00: [-0.012084, 0.012084]}

# coefficients of exponentially decrease beta with epoch
# y=a*(0.5^(0.01*(x-1)))+c, each key-value represent "epoch_ratio: [a, c]"
decrease_coe_dict = {0.10: [0.099211, -0.093211],
                     0.20: [0.048625, -0.042625],
                     0.30: [0.032949, -0.026949],
                     0.40: [0.025330, -0.019330],
                     0.50: [0.020835, -0.014835],
                     0.60: [0.017875, -0.011875],
                     0.70: [0.015783, -0.009783],
                     0.80: [0.014230, -0.008230],
                     0.90: [0.013033, -0.007033],
                     1.00: [0.012084, -0.006084]}


def update_beta_with_epoch(args, cur_epoch):
    '''update beta value based on current epoch number'''
    beta_min, beta_max = 0.0, args.beta_opt
    if args.cl_strategy == 'beta_increase' or args.cl_strategy == 'beta_decrease':
        curri_last_epoch = int(args.n_epochs * args.epoch_ratio)
        beta_min, beta_max = 0.0, args.beta_opt
        beta_step = (beta_max - beta_min) / (curri_last_epoch - 1)
    elif args.cl_strategy == 'beta_increase_exp' or args.cl_strategy == 'beta_decrease_exp':
        beta = 0

    if args.cl_strategy == 'beta_increase':
        beta = beta_min + (cur_epoch - 1) * beta_step
        if beta > beta_max:
            beta = beta_max
    elif args.cl_strategy == 'beta_decrease':
        beta = beta_max - (cur_epoch - 1) * beta_step
        if beta < beta_min:
            beta = beta_min
    elif args.cl_strategy == 'beta_increase_exp':
        beta = increase_coe_dict[args.epoch_ratio][0] * \
            pow(0.5, 0.01 * (cur_epoch - 1)) + increase_coe_dict[args.epoch_ratio][1]
        if beta > beta_max:
            beta = beta_max
    elif args.cl_strategy == 'beta_decrease_exp':
        beta = decrease_coe_dict[args.epoch_ratio][0] * \
            pow(0.5, 0.01 * (cur_epoch - 1)) + decrease_coe_dict[args.epoch_ratio][1]
        if beta < beta_min:
            beta = beta_min
    return beta


def train(args, model, criterion, train_loader, valid_loader, validation, optimizer, model_path):
    valid_losses = []
    save_model_path = model_path
    best_iou, best_dice, best_epoch_iou, best_epoch_dice = 0.0, 0.0, 0, 0
    start_epoch = 0

    print('==> Training started.')
    for epoch in range(start_epoch + 1, args.n_epochs + 1):
        model.train()
        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}'.format(epoch, optimizer.param_groups[0]['lr']))

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            batch_size = inputs.size(0)
            loss.backward()
            optimizer.step()

            tq.update(batch_size)
        tq.close()

        # ========================== Validation ========================== #
        valid_metrics = validation(args, model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        valid_iou = valid_metrics['iou']
        valid_dice = valid_metrics['dice']

        checkpoint = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_iou": valid_iou,
            "best_iou": best_iou,
            "best_epoch_iou": best_epoch_iou,
            "best_epoch_dice": best_epoch_dice,
            "valid_loss": valid_loss
        }

        if valid_dice > best_dice:
            best_dice = valid_dice
            best_epoch_dice = epoch
            print('===================saving the best model!========================')
            torch.save(checkpoint, os.path.join(save_model_path, "best_model_dice.pt"))

        print('Best epoch for dice unitl now:', best_epoch_dice, ' best dice:', best_dice)
    print('==> Training finished.')
    print('Best epoch for dice:', best_epoch_dice, ' best dice:', best_dice)


def train_cl(args, model, criterion, train_image_paths, val_image_paths, validation, optimizer, model_path):
    valid_losses = []
    save_model_path = model_path
    best_iou, best_dice, best_epoch_iou, best_epoch_dice = 0.0, 0.0, 0, 0
    start_epoch = 0 

    print('==> Training started.')
    for epoch in range(start_epoch + 1, args.n_epochs + 1):
        model.train()
        args.beta = update_beta_with_epoch(args, epoch)

        if args.AM == 'True':
            train_loader = make_loader(args, train_image_paths, shuffle=True, transform=train_transform_AM(p=1, im_size = args.img_size),
                                       batch_size=args.batch_size, mode='train')
        else:
            train_loader = make_loader(args, train_image_paths, shuffle=True, transform=train_transform(p=1, im_size = args.img_size),
                                       batch_size=args.batch_size, mode='train')
        valid_loader = make_loader(args, val_image_paths, transform=val_transform(p=1, im_size = args.img_size),
                                    batch_size=args.batch_size, mode='val')

        tq = tqdm.tqdm(total=(len(train_loader) * args.batch_size))
        tq.set_description('Epoch {}, lr {}, beta {}'.format(epoch, optimizer.param_groups[0]['lr'], args.beta))

        for _, (inputs, targets) in enumerate(train_loader):
            inputs = cuda(inputs)
            with torch.no_grad():
                targets = cuda(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            batch_size = inputs.size(0)
            loss.backward()
            optimizer.step()

            tq.update(batch_size)
        tq.close()

        # ========================== Validation ========================== #
        valid_metrics = validation(args, model, criterion, valid_loader)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        valid_iou = valid_metrics['iou']
        valid_dice = valid_metrics['dice']

        checkpoint = {
            "net": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_iou": valid_iou,
            "best_iou": best_iou,
            "best_epoch_iou": best_epoch_iou,
            "best_epoch_dice": best_epoch_dice,
            "valid_loss": valid_loss
        }

        if valid_dice > best_dice:
            best_dice = valid_dice
            best_epoch_dice = epoch
            print('===================saving the best model!========================')
            torch.save(checkpoint, os.path.join(save_model_path, "best_model_dice.pt"))

        print('Best epoch for dice unitl now:', best_epoch_dice, ' best dice:', best_dice)
    print('==> Training finished.')
    print('Best epoch for dice:', best_epoch_dice, ' best dice:', best_dice)
