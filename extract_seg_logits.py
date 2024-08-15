import argparse
import os, torch
import h5py, glob, time
import numpy as np
from PIL import Image
from pathlib import Path
import dinov2.eval.segmentation_m2f.models.segmentors
from mmseg.apis import init_segmentor
import mmcv, urllib
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor
import dinov2.eval.segmentation.models
from PIL import Image
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose
import numpy as np
import dinov2.eval.segmentation.utils.colormaps as colormaps
import cv2
from mmseg.apis import inference_segmentor
from my_mask2former_interface import render_segmentation


DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
DATASET_LABELS = {
    "ade20k": colormaps.ADE20K_CLASS_NAMES,
    "voc2012": colormaps.VOC2012_CLASS_NAMES,
}

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}

CONFIG_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f_config.py"
CHECKPOINT_URL = f"{DINOV2_BASE_URL}/dinov2_vitg14/dinov2_vitg14_ade20k_m2f.pth"


def parse_args():
    parser = argparse.ArgumentParser(description='Extract segmentation logits')
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Database directory")
    parser.add_argument("--set-name", type=str, required=True,
                        help="Set name")
    args = parser.parse_args()

    return args


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def load_mask2former():
    cfg_str = load_config_from_url(CONFIG_URL)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    model = init_segmentor(cfg)
    load_checkpoint(model, CHECKPOINT_URL, map_location="cpu")
    model.cuda()
    model.eval()
    return model

def render_segmentation(seg_cls, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    clsmap = DATASET_LABELS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    clsmap_array = np.array(clsmap)
    seg_mask = colormap_array[seg_cls + 1]
    seg_mask_cls = clsmap_array[seg_cls + 1]
    return Image.fromarray(seg_mask), seg_mask_cls

def main(args):
    set_dir = os.path.join(args.base_dir, args.set_name)

    save_dir = os.path.join(set_dir, 'lwir_seg')
    os.makedirs(save_dir, exist_ok=True)

    image_list = glob.glob(os.path.join(set_dir, 'lwir/*.png'))    

    model = load_mask2former()

    start_time = time.time()
    for i, image_path in enumerate(image_list):
        print("{}/{} image: {}".format(i, len(image_list), image_path))
        
        im = Image.open(image_path)
        im = im.convert("RGB")

        array = np.array(im)[:, :, ::-1] # BGR
        seg_logits = inference_segmentor(model, array)[0]
        seg_cls = seg_logits.argmax(dim=0)
        seg_cls = seg_cls.cpu().numpy()
        seg_logits = seg_logits.cpu().numpy()
        seg_mask, seg_cls = render_segmentation(seg_cls, "ade20k")

        # save dir
        save_name = Path(image_path).stem
        seg_mask.save(os.path.join(save_dir, save_name + "_mask.png"))
        np.save(os.path.join(save_dir, save_name + "_logits.npy"), seg_logits)

        if i % 100 == 1:
            print("\t ===> Elapsed time = %.2fs" % (time.time() -start_time))


    print("Done!!!")

if __name__ == "__main__":
    args = parse_args()
    main(args)