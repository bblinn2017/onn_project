import configargparse

def train_config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--exp_name", type=str,
                        help='experiment name')
    parser.add_argument("--base_dir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--data_dir", type=str, default=None,
                        help='ref data directory')
    parser.add_argument("--datalist", type=str, default='./data/renderppl-A',
                        help='data split of train, validation and testing')
    parser.add_argument("--progress_refresh_rate", type=int, default=1,
                        help='how many iterations to show psnrs or iters')
    parser.add_argument("--iter", type=int, default=10,
                        help='number of iterations')
    parser.add_argument("--batch_size", type=int, default=1,
                        help='batch size')
    parser.add_argument("--save_checkpoint_freq", type=int, default=100)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    parser.add_argument("--rgb_weight", type=float, default=1.)
    parser.add_argument("--depth_weight", type=float, default=1.)
    parser.add_argument("--normal_weight", type=float, default=1.)
    parser.add_argument("--mask_weight", type=float, default=1.)
    parser.add_argument("--num_ref_views", type=int, default=16)
    parser.add_argument("--num_tar_views", type=int, default=32)
    
    parser.add_argument("--lr_nerf_renderer", type=float, default=0.0005,
                        help='learning rate for nerf renderer')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help='learning rate decay iterations')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='target learning rate decay ratio')

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()

def test_config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--save_dir", type=str, required=True,
                        help='experiment name')
    parser.add_argument("--data_dir", type=str, default='/home/ICT2000/lshichen/Desktop/mnt/renderppl',
                        help='input data directory')
    parser.add_argument("--test_data_dir", type=str,
                        help='input test data directory')
    parser.add_argument("--datalist", type=str, default='./data/renderppl-A',
                        help='data split of train, validation and testing')
    parser.add_argument("--ref_cam_dir", type=str, default='/home/ICT2000/hxiao/Dataset/renderppl-16lightstageCam',
                        help='reference camera directory')
    parser.add_argument("--test_cam_dir", type=str, default='/home/ICT2000/hxiao/Dataset/renderppl-manual60Cams-forTest',
                        help='test camera directory')
    parser.add_argument("--num_ref_views", type=int, default=16)
    parser.add_argument("--num_tar_views", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--load_checkpoint", type=str, default=None)

    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()
