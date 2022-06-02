import random

import cv2
import numpy as np
import torch
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import img_to_tensor
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from augmentations import augmentations_all, augmentations_H, augmentations_L
from prepare_data import get_data_paths_list


class FundusDataset(Dataset):
    def __init__(self, args, file_names, transform=None, mode='train'):
        self.args = args
        self.file_names = file_names
        self.transform = transform
        self.mode = mode
        self.td_image_paths = get_data_paths_list(domain='Domain2', split='train', type='image')

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_file_name = self.file_names[idx]
        image = load_image(img_file_name)
        mask = load_mask(img_file_name)

        if self.mode == 'train' or self.mode == 'val':
            if self.args.method == 'FDA':
                selected_td_image_path = self.td_image_paths[np.random.randint(len(self.td_image_paths))]
                selected_td_image = load_image(selected_td_image_path)
                if self.args.beta_random == 'True':
                    image = fourier_image_perturbation(image, selected_td_image, beta=random.uniform(0.0, 0.006), ratio=1.0)
                elif self.args.ratio_random == 'True':
                    image = fourier_image_perturbation(image, selected_td_image, beta=1.0, ratio=random.uniform(0.0, 1.0))    
                else:
                    image = fourier_image_perturbation(image, selected_td_image, beta=self.args.beta, ratio=self.args.ratio)

            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]
            if self.mode != 'val' and self.args.method == 'FDA' and self.args.AM == 'True':
                return Image.fromarray(image), torch.from_numpy(mask).long()
            else:
                return img_to_tensor(image), torch.from_numpy(mask).long()
        elif self.mode == 'eva':
            data = {"image": image, "mask": mask}
            augmented = self.transform(**data)
            image, mask = augmented["image"], augmented["mask"]
            return img_to_tensor(image), torch.from_numpy(mask).long()
        else:
            return img_to_tensor(image), str(img_file_name)


def load_image(path):
    img = cv2.imread(str(path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_mask(path):
    mask = cv2.imread(str(path).replace('image', 'mask').replace('jpg', 'png'), 0)
    mask[mask==0], mask[mask==128], mask[mask==255] = 2, 1, 0 # cup, disc, background
    return mask


def train_transform(p=1, im_size=384):
    return Compose([
        Resize(im_size, im_size, always_apply=True, p=1),
        Normalize(p=1)
    ], p=p)


def val_transform(p=1, im_size=384):
    return Compose([
        Resize(im_size, im_size, always_apply=True, p=1),
        Normalize(p=1)
    ], p=p)


def test_transform(p=1, im_size=384):
    return Compose([
        Resize(im_size, im_size, always_apply=True, p=1),
        Normalize(p=1)
    ], p=p)

def train_transform_AM(p=1, im_size=384):
    return Compose([
        Resize(im_size, im_size, always_apply=True, p=1),
    ], p=p)

_IMG_MEAN, _IMG_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # same as train_transform()

preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_IMG_MEAN, _IMG_STD)
    ])


def make_loader(args, file_names, shuffle=False, transform=None, batch_size=1, mode='train'):
    dataset=FundusDataset(args, file_names, transform=transform, mode=mode)

    if args.AM == 'True' and mode == 'train':       
        dataset = AugMixData(dataset, preprocess, im_size = args.img_size, level=args.AM_level)

    return DataLoader(
        dataset=dataset,
        shuffle=shuffle,
        num_workers=args.workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )

def fourier_transform(image_np):
    '''perform 2D-fft of the input image and return amplitude and phase component'''
    # image.shape: H, W, C
    fft_image_np = np.fft.fft2(image_np, axes=(0, 1))
    # extract amplitude and phase of both ffts
    amp_np, pha_np = np.abs(fft_image_np), np.angle(fft_image_np)
    return amp_np, pha_np

def amplitude_mutate(amp1, amp2, beta=0.001, ratio=1.0):
    amp1_ = np.fft.fftshift(amp1, axes=(0, 1))
    amp2_ = np.fft.fftshift(amp2, axes=(0, 1))

    h, w, c = amp1_.shape
    h_crop = int(h * beta)
    w_crop = int(w * beta)
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    amp1_c = np.copy(amp1_)
    amp2_c = np.copy(amp2_)
    amp1_[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        ratio * amp2_c[h_start:h_start + h_crop, w_start:w_start + w_crop] \
        + (1 - ratio) * amp1_c[h_start:h_start + h_crop, w_start:w_start + w_crop]

    amp1_ = np.fft.ifftshift(amp1_, axes=(0, 1))
    return amp1_


def fourier_image_perturbation(image1_np, image2_np, beta=0.001, ratio=0.0):
    '''perturb image via mutating the amplitude component'''
    amp1_np, pha1_np = fourier_transform(image1_np)
    amp2_np, pha2_np = fourier_transform(image2_np)
    # mutate the amplitude part of source with target
    amp_src_ = amplitude_mutate(amp1_np, amp2_np, beta=beta, ratio=ratio)

    # mutated fft of source
    fft_image1_ = amp_src_ * np.exp(1j * pha1_np)

    # get the mutated image
    image12 = np.fft.ifft2(fft_image1_, axes=(0, 1))
    image12 = np.real(image12)
    image12 = np.uint8(np.clip(image12, 0, 255))

    return image12


class AugMixData(torch.utils.data.Dataset):
    def __init__(self, dataset, preprocess, im_size, js_loss=False, n_js=3, level=3, alpha=1, mixture_width=3, mixture_depth=0):
        self.dataset = dataset
        self.preprocess = preprocess
        self.js_loss = js_loss
        self.n_js = n_js
        self.level = level
        self.alpha = alpha
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.im_size = im_size

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if self.js_loss:
            xs = [self.preprocess(x), self.augmix(x)]
            while len(xs) < self.n_js:
                xs.append(self.augmix(x))
            return xs, y
        else:
            return self.augmix(x), y

    def __len__(self):
        return len(self.dataset)

    def augmix(self, img):
        aug_list = augmentations_L
        ws = np.float32(np.random.dirichlet([self.alpha] * self.mixture_width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        mixed_image = torch.zeros_like(self.preprocess(img))

        for i in range(self.mixture_width):
            aug_img = img.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(1, 4)
            for d in range(depth):
                op = np.random.choice(aug_list)
                aug_img = op(aug_img, self.level, self.im_size)
            mixed_image += ws[i] * self.preprocess(aug_img)
        return m * self.preprocess(img) + (1-m) * mixed_image
