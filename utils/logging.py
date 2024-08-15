import torch
import shutil
import os
import torchvision.utils as tvu
import argparse


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
    
def print_namespace(namespace, indent=0):
    for key, value in vars(namespace).items():
        if isinstance(value, argparse.Namespace):
            print("  " * indent + f"{key}:")
            print_namespace(value, indent + 1)
        else:
            print("  " * indent + f"{key}: {value}")
