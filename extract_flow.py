import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.utils.misc import process_cfg
from utils import flow_viz

from core.Networks import build_network

from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
import itertools
import imageio
from pathlib import Path


def prepare_image(base_dir, txt):
    print(f"preparing image...")
    print(f"Input image sequence dir = {base_dir}")
    train_list = os.path.join(base_dir, txt)
    print("==> load image list = {}".format(train_list))
    with open(train_list) as f:
        contents = f.readlines()
        image_list = [i.strip() for i in contents]
    images = [];
    ids = [];
    for fn in image_list:
        print(f"Load {fn}")
        id = Path(fn).stem
        img = Image.open(os.path.join(base_dir, fn+".png"))
        img = img.convert("RGB")
        img = np.array(img).astype(np.uint8)[..., :3]
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        images.append(img)
        ids.append(id)
    
    return torch.stack(images), ids



@torch.no_grad()
def MOF_inference(model, input_images):

    model.eval()

    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

@torch.no_grad()
def BOF_inference(model, input_images):

    model.eval()

    input_images = input_images[None].cuda()
    padder = InputPadder(input_images.shape)
    input_images = padder.pad(input_images)
    flow_pre, _ = model(input_images, {})
    flow_pre = padder.unpad(flow_pre[0]).cpu()

    return flow_pre

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_flows(flow_pre, save_dir, ids):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    N = flow_pre.shape[0]

    for idx in range(N//2):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{}_to_{}.png'.format(save_dir, ids[idx+1], ids[idx+2]))
        flow_pre_np = flow_pre[idx].permute(1, 2, 0).numpy()
        np.save('{}/flow_{}_to_{}.npy'.format(save_dir, ids[idx+1], ids[idx+2]), flow_pre_np)

    for idx in range(N//2, N):
        flow_img = flow_viz.flow_to_image(flow_pre[idx].permute(1, 2, 0).numpy())
        image = Image.fromarray(flow_img)
        image.save('{}/flow_{}_to_{}.png'.format(save_dir, ids[idx-N//2+1], ids[idx-N//2]))
        flow_pre_np = flow_pre[idx].permute(1, 2, 0).numpy()
        np.save('{}/flow_{}_to_{}.npy'.format(save_dir, ids[idx-N//2+1], ids[idx-N//2]), flow_pre_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='BOF')
    parser.add_argument('--base-dir', default='')
    parser.add_argument('--set-name', default='')
    parser.add_argument('--out-dir', default='')
    args = parser.parse_args()

    if args.mode == 'MOF':
        from configs.multiframes_sintel_submission import get_cfg
    elif args.mode == 'BOF':
        from configs.sintel_submission import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))

    model = torch.nn.DataParallel(build_network(cfg))
    model.load_state_dict(torch.load(cfg.model))

    model.cuda()
    model.eval()

    print(cfg.model)
    print("Parameter Count: %d" % count_parameters(model))

    input_images, ids = prepare_image(base_dir=cfg.base_dir, txt=cfg.set_name + ".txt")
    n = len(input_images)

    start_time = time.time()
    for i in range(n-2):
        print(f"\nProcess {i}/{n}")
        triplet = torch.clone(input_images[i:i+3])
        triplet_ids = ids[i:i+3]
        print(triplet_ids)
        with torch.no_grad():
            if args.mode == 'MOF':
                from configs.multiframes_sintel_submission import get_cfg
                flow_pre = MOF_inference(model=model.module, input_images=triplet)
            elif args.mode == 'BOF':
                from configs.sintel_submission import get_cfg
                flow_pre = BOF_inference(model.module, input_images=triplet)

        save_flows(flow_pre=flow_pre, 
                   save_dir=os.path.join(cfg.base_dir, cfg.set_name, 'lwir_flow'), 
                   ids=triplet_ids)
        if i % 100 == 1:
            print("\t Elapsed time = %.2fs" % (time.time()-start_time))

