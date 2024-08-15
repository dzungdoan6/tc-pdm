import argparse
import os
import yaml
import torch
import torch.utils.data
import numpy as np
import datasets
from models import DenoisingDiffusion
from torch.distributed import init_process_group, destroy_process_group
from utils.logging import print_namespace


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Training Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps for validation image patches")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored validation image patches")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    parser.add_argument('--txt', default='', type=str, required=True,
                    help='txt file')  
    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def main():
    args, config = parse_args_and_config()

    ddp_setup()
    # setup device to run
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print("Using device: {}".format(device))
    # config.device = device

    # print config
    print("\n")
    print_namespace(config)
    print("\n")
    
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)

    # create model
    print("=> creating denoising-diffusion model...")
    diffusion = DenoisingDiffusion(args, config, phase='train')
    diffusion.train(DATASET)
    destroy_process_group()


if __name__ == "__main__":
    main()
