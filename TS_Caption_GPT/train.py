# train.py
import argparse
import torch
import os
from exp.exp import Exp_time_caption

os.environ['CUDA_VISIBLE_DEVICE']='0,1'

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./datasets')
    parser.add_argument('--data_path', type=str, default='timecaption.json')
    parser.add_argument('--data', type=str, default='timecaption')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    
    parser.add_argument('--n_heads_cross', type=int, default=8, help='n_heads_cross')
    parser.add_argument('--num_layer_cross', type=int, default=8, help='number cross attention')
    
    parser.add_argument('--input_dim', type=int, default=12, help='input dim')
    parser.add_argument('--time_dim', type=int, default=256, help='time embedding dim')
    parser.add_argument('--text_dim', type=int, default=768, help='text embedding dim')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--itr', type=int, required=True, default=1, help='status')
    parser.add_argument('--max_length', type=int, default=40, help='max length')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')


    args = parser.parse_args()
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    if args.is_training == 1:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp_time_caption(args)  # set experiments
            setting = '{}_batch_size{}_epochs{}_dim{}'.format(args.data, args.batch_size, args.epochs, args.text_dim)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    elif args.is_training == 0:
        ii = 0
        setting = '{}_batch_size{}_epochs{}_dim{}'.format(args.data, args.batch_size, args.epochs, args.text_dim)

        exp = Exp_time_caption(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
    else:
        setting = '{}_batch_size{}_epochs{}_dim{}'.format(args.data, args.batch_size, args.epochs, args.text_dim)
        exp = Exp_time_caption(args)
        x = [[1.0000, 0.3676, 0.1830, 0.0182, 0.0000, 0.7867, 0.5326, 0.0847, 0.7080, 0.7594, 0.1922, 0.7247], [1.0000, 0.3676, 0.1830, 0.0182, 0.0000, 0.7867, 0.5326, 0.0847, 0.7080, 0.7594, 0.1922, 0.7247]]
        x_tensor = torch.tensor(x, dtype=torch.float32)

        # x_min = torch.min(x_tensor, dim=1, keepdim=True)[0]
        # x_max = torch.max(x_tensor, dim=1, keepdim=True)[0]

        # x_norm = (x_tensor - x_min) / (x_max - x_min)
        x_norm = x_tensor.unsqueeze(-1)

        y = exp.generate(x_norm, max_length=40, test=args.is_training, setting=setting)

        print(y)