#----- Example training commands -----#

### Vanilla ###
python3 train.py --model UNet --save_model_path saved_model/Vanilla/UNet # UNet
python3 train.py --model SWin-UNet --save_model_path saved_model/Vanilla/SWin-UNet # SWin-UNet


### FDA ###
python3 train.py --method FDA --model UNet --ratio 1.0 --beta 0.006 --save_model_path saved_model/FDA/UNet # UNet
python3 train.py --method FDA --model SWin-UNet --ratio 1.0 --beta 0.006 --save_model_path saved_model/FDA/SWin-UNet # SWin-UNet

### Curri-FDA ###
python3 train.py --method FDA --model UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase --epoch_ratio 0.5 --save_model_path saved_model/Curri-FDA/UNet/lin-inc # UNet, lin-inc
python3 train.py --method FDA --model UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase_exp --epoch_ratio 0.5 --save_model_path saved_model/Curri-FDA/UNet/exp-inc # UNet, exp-inc

python3 train.py --method FDA --model SWin-UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase --epoch_ratio 0.5 --save_model_path saved_model/Curri-FDA/SWin-UNet/lin-inc # SWin-UNet, lin-inc
python3 train.py --method FDA --model SWin-UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase_exp --epoch_ratio 0.5 --save_model_path saved_model/Curri-FDA/SWin-UNet/exp-inc # SWin-UNet, exp-inc

### Curri-AFDA ###
python3 train.py --method FDA --model UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase --epoch_ratio 0.5 --AM True --AM_level 3 --save_model_path saved_model/Curri-AFDA/UNet/lin-inc/AM-3 # UNet, lin-inc, AM-3
python3 train.py --method FDA --model UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase_exp --epoch_ratio 0.5 --AM True --AM_level 3 --save_model_path saved_model/Curri-AFDA/UNet/exp-inc/AM-3 # UNet, exp-inc, AM-3

python3 train.py --method FDA --model SWin-UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase --epoch_ratio 0.5 --AM True --AM_level 3 --save_model_path saved_model/Curri-AFDA/SWin-UNet/lin-inc/AM-3 # SWin-UNet, lin-inc, AM-3
python3 train.py --method FDA --model SWin-UNet --ratio 1.0 --beta_opt 0.006 --curriculum True --cl_strategy beta_increase_exp --epoch_ratio 0.5 --AM True --AM_level 3 --save_model_path saved_model/Curri-AFDA/SWin-UNet/exp-inc/AM-3 # SWin-UNet, exp-inc, AM-3
