import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='OepfTSF')


def add_arg(*name_or_flags, **kwargs):
    existing = set()
    for act in parser._actions:
        for opt in act.option_strings:
            existing.add(opt)
    for f in name_or_flags:
        if isinstance(f, str) and f.startswith("--") and f in existing:
            return
    parser.add_argument(*name_or_flags, **kwargs)


# basic config
add_arg('--is_training', type=int, default=1, help='status')
add_arg('--train_only', type=bool, required=False, default=False)
add_arg('--model_id', type=str, default='ETTh1_96_ema', help='model id')
add_arg('--model', type=str, default='OepfTSF', help='[interPDN, DLinear]')

# DLinear
add_arg('--individual', action='store_true', default=False)

# data loader
add_arg('--data', type=str, default='ETTh1')
add_arg('--root_path', type=str, default='./dataset')
add_arg('--data_path', type=str, default='ETTh1.csv')
add_arg('--features', type=str, default='M', help='[M, S, MS]')
add_arg('--target', type=str, default='OT')
add_arg('--freq', type=str, default='h')
add_arg('--checkpoints', type=str, default='./Checkpoints/')
add_arg('--embed', type=str, default='timeF')

# forecasting task
add_arg('--seq_len', type=int, default=512)
add_arg('--label_len', type=int, default=48)
add_arg('--pred_len', type=int, default=96)
add_arg('--enc_in', type=int, default=7)

# patching
add_arg('--patch_len', type=int, default=16)
add_arg('--stride', type=int, default=8)
add_arg('--padding_patch', default='end')

# moving average
add_arg('--ma_type', type=str, default='ema', help='reg, ema, dema')
add_arg('--alpha', type=float, default=0.3)
add_arg('--beta', type=float, default=0.3)

# optimization
add_arg('--num_workers', type=int, default=0)
add_arg('--itr', type=int, default=1)
add_arg('--train_epochs', type=int, default=100)
add_arg('--batch_size', type=int, default=1024)
add_arg('--patience', type=int, default=10)
add_arg('--learning_rate', type=float, default=0.0001)
add_arg('--des', type=str, default='Exp')
add_arg('--loss', type=str, default='mse')
add_arg('--lradj', type=str, default='sigmoid')

# AMP
add_arg('--use_amp', action='store_true', default=False)
add_arg('--amp_dtype', type=str, default='bf16', choices=['fp16', 'bf16'])

add_arg('--revin', type=int, default=1)

# loss weights
add_arg('--con_cls_1', type=float, default=0.05)
add_arg('--con_cls_2', type=float, default=0.05)
add_arg('--con_time', type=float, default=0.1)

# accuracy-oriented configs
add_arg('--horizon_weighting', type=str, default='atan', choices=['atan', 'sqrt', 'linear', 'none'])

add_arg('--train_loss_type', type=str, default='mix', choices=['wmae', 'wmse', 'whuber', 'mix'])
add_arg('--mix_mse_weight', type=float, default=0.85)
add_arg('--huber_delta', type=float, default=1.0)
add_arg('--grad_clip', type=float, default=0.0)

add_arg('--dist_sup_weight', type=float, default=0.25)
add_arg('--dist_warmup_epochs', type=int, default=10)
add_arg('--label_sigma_ratio', type=float, default=1.0)
add_arg('--sigma_anneal_from', type=float, default=1.4)
add_arg('--sigma_anneal_to', type=float, default=1.0)

add_arg('--support_lo', type=float, default=-4.0)
add_arg('--support_hi', type=float, default=4.0)
add_arg('--support_tau', type=float, default=None)
add_arg('--tau_anneal_from', type=float, default=None)
add_arg('--tau_anneal_to', type=float, default=None)

add_arg('--support_reg_weight', type=float, default=0.1)
add_arg('--support_warmup_epochs', type=int, default=10)
add_arg('--support_min_gap_ratio', type=float, default=0.25)
add_arg('--support_smooth_weight', type=float, default=0.1)
add_arg('--support_points_path', type=str, default='')

# GPU
add_arg('--use_gpu', type=bool, default=True)
add_arg('--gpu', type=int, default=0)
add_arg('--use_multi_gpu', action='store_true', default=False)
add_arg('--devices', type=str, default='0')
add_arg('--test_flop', action='store_true', default=False)

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

if getattr(args, "support_tau", None) is not None:
    try:
        args.support_tau = float(args.support_tau)
    except Exception:
        args.support_tau = None

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
            args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.des, ii
        )
        exp = Exp(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
        args.model_id, args.model, args.data, args.features,
        args.seq_len, args.label_len, args.pred_len, args.des, ii
    )
    exp = Exp(args)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
