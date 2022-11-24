# Curri-AFDA

## Introduction

This is the Pytorch implementation for the paper: 

**Curriculum-Based Augmented Fourier Domain Adaption for Robust Medical Image Segmentation**

by [An Wang](wa09@link.cuhk.edu.hk)$^\dagger$, [Mobarakol Islam](m.islam20@imperial.ac.uk)$^\dagger$, [Mengya Xu](mengya@u.nus.edu)$^\dagger$, and [Hongliang Ren](ren@nus.edu.sg)

$\dagger$: equal contribution


![curri-afda-overall](img/curri-afda-overall.jpeg?raw=true "curri-afda-overall")
This work proposes the Curriculum-based Augmented Fourier Domain Adaptation (Curri-AFDA) to achieve impressive adaptation, generalization, and robustness performance for the medical image segmentation task.

## Environment
- NVIDIA RTX3090
- Python 3.8
- Pytorch 1.10
- Conda3
- Check [environment.yml](https://github.com/lofrienger/Curri-AFDA/blob/main/environment.yml) for more dependencies.

## Usage
### Dataset
1. The Retina datasets used in this repository can be downloaded from [Dofe-Fundus](https://github.com/emma-sjwang/Dofe). We follow their data splits and use the ROI images.
2. The Nuclei datasets used in our work are all publicly available. 

### Train/Test
1. (Optional) For the SWin-UNet backbone, get the [pretrained_model](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth) from [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) and put into the folder [pretrained_ckpt](https://github.com/lofrienger/Curri-AFDA/tree/main/pretrained_ckpt).
2. Update your Retina dataset path in [prepare_data.py](https://github.com/lofrienger/Curri-AFDA/blob/main/prepare_data.py).
3. Example training commands

- Vanilla
  
`python3 train.py --model UNet --save_model_path saved_model/Vanilla/UNet # UNet`

`python3 train.py --model SWin-UNet --save_model_path saved_model/Vanilla/SWin-UNet # SWin-UNet`

- FDA 
  
`python3 train.py --method FDA --model UNet --ratio 1.0 --beta 0.006 --save_model_path saved_model/FDA/UNet # UNet`

`python3 train.py --method FDA --model SWin-UNet --ratio 1.0 --beta 0.006 --save_model_path saved_model/FDA/SWin-UNet # SWin-UNet`

- Curri-FDA

`python3 train.py --method FDA --model UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase --epoch_ratio 0.5 --save_model_path saved_model/Curri-FDA/UNet/lin-inc # UNet, lin-inc`


- Curri-AFDA

`python3 train.py --method FDA --model UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase --epoch_ratio 0.5 --AM True --AM_level 3 --save_model_path saved_model/Curri-AFDA/UNet/lin-inc/AM-3 # UNet, lin-inc, AM-3`

More training commands are available in [exp.sh](https://github.com/lofrienger/Curri-AFDA/blob/main/exp.sh).

## Acknowledgement
Some of the codes are adapted from [FedDG](https://github.com/liuquande/FedDG-ELCFS), [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet), and [robot-surgery-segmentation](https://github.com/ternaus/robot-surgery-segmentation).
## Citation
```
@misc{dummy,
      title={Curriculum-Based Augmented Fourier Domain Adaption for Robust Medical Image Segmentation}
}
```
