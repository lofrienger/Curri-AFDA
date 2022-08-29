import argparse
import os
import warnings
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import Adam

from config import get_config
from models import UNet
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import *
from validation import validation_multi

warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--batch_size', type=int, default=16)
    arg('--n_epochs', type=int, default=100, help='the number of total epochs')
    arg('--lr', type=float, default=1e-3)
    arg('--workers', type=int, default=4)
    arg('--seed', type=int, default=3407)
    arg('--num_classes', type=int, default=3)
    arg('--beta', type=float, default=0.006, help='the amplitude scaling coefficient')
    arg('--ratio', type=float, default=1, help='the weighting coefficient')
    arg('--model', type=str, default='UNet', choices=['UNet', 'SWin-UNet'])
    arg('--method', type=str, default='Vanilla', choices=['Vanilla', 'FDA'])
    arg('--AM', type=str, default='False', help='use augmentation mixing or not')
    arg('--AM_level', type=int, default=3, help='AM level')
    arg('--curriculum', type=str, default='False', help='use curriculum or not')
    arg('--beta_random', type=str, default='False', help='beta is rondom[0, 0.006] or not, ratio is 1.0')
    arg('--ratio_random', type=str, default='False', help='ratio is rondom[0, 1] or not, beta is 1.0')
    arg('--cl_strategy', type=str, default='None', help='curriculum learning strategy')
    arg('--epoch_ratio', type=float, default=0.4, help='the ratio of epochs for curriculum')
    arg('--save_model_path', type=str, default='saved_model/')
    arg('--beta_opt', type=float, default=0.006, help='optimal beta')

    arg('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', help='path to swin-unet config file')
    arg('--img_size', type=int, default=384, help='input patch size of network input')

    args = parser.parse_args()

    # The results are reproduciable, deterministic training
    seed_everything(args.seed)

    # initialize model
    if args.model == 'UNet':
        model = UNet(num_classes=args.num_classes).cuda()
    elif args.model == 'SWin-UNet':
        config = get_config(args)
        model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()
        model.load_from(config)

        args.img_size = 224

    # assign GPU device
    if torch.cuda.is_available():
        num_gpu = torch.cuda.device_count()
        print('The total available GPU_number:', num_gpu)
        if num_gpu > 1:  # has more than 1 gpu
            device_ids = np.arange(num_gpu).tolist()
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
    else:
        raise SystemError('GPU device not found')

    # create saved model path
    model_path = args.save_model_path
    Path(model_path).mkdir(parents=True, exist_ok=True)

    # ================ optimizer  ======================= #
    optimizer = Adam(model.parameters(), lr=args.lr)

    print('=============== Print Args ======================')
    pprint(vars(args))
    print('=================================================')


    # get train, val, test files
    train_image_paths = get_data_paths_list(
        domain='Domain3', split='train', type='image')
    val_image_paths = get_data_paths_list(
        domain='Domain3', split='test', type='image')
    test_image_paths = get_data_paths_list(
        domain='Domain2', split='test', type='image')
    print('Num train = {}, Num_val = {}, Num test = {}'.format(
        len(train_image_paths), len(val_image_paths), len(test_image_paths)))

    if args.method == 'Vanilla' or (args.method == 'FDA' and args.curriculum == 'False'):
        if args.AM == 'True':
            transform_train = train_transform_AM(p=1, im_size = args.img_size)
        else:
            transform_train = train_transform(p=1, im_size = args.img_size)
        train_loader = make_loader(args, train_image_paths, shuffle=True, transform=transform_train,
                                   batch_size=args.batch_size, mode='train')
        valid_loader = make_loader(args, val_image_paths, transform=val_transform(p=1, im_size = args.img_size),
                                   batch_size=args.batch_size, mode='val')
        train(args=args,
              model=model,
              criterion=nn.CrossEntropyLoss(),
              train_loader=train_loader,
              valid_loader=valid_loader,
              validation=validation_multi,
              optimizer=optimizer,
              model_path=model_path
              )
    elif args.method == 'FDA' and args.curriculum == 'True':                          
        train_cl(
            args=args,
            model=model,
            criterion=nn.CrossEntropyLoss(),
            train_image_paths=train_image_paths,
            val_image_paths=val_image_paths,
            validation=validation_multi,
            optimizer=optimizer,
            model_path=model_path
        )
    else:
        raise Exception("** Invalid training settings!!! **")

    print('==> Testing started.')
    test_loader = make_loader(args, test_image_paths, transform=test_transform(p=1, im_size = args.img_size),
                                  batch_size=args.batch_size, mode='eva')
    checkpoint = torch.load(os.path.join(model_path, 'best_model_dice.pt'))
    model.load_state_dict(checkpoint['net'])  # load the best model
    metrics = validation_multi(
        args, model=model, criterion=nn.CrossEntropyLoss(), valid_loader=test_loader)
    print('==> Testing finished.')  

if __name__ == '__main__':
    main()
