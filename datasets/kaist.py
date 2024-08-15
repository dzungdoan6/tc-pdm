import os
import torch
import numpy as np
import torchvision
import torch.utils.data
from PIL import Image
from pathlib import Path
import random
from torch.utils.data.distributed import DistributedSampler

class KAIST:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def get_train_loader(self, txt):
        train_dataset = KAIST_train(dir = os.path.join(self.config.data.data_dir, self.config.data.dataset),
                                    n=self.config.training.patch_n,
                                    patch_size=self.config.data.image_size,
                                    transforms=self.transforms,
                                    filelist=txt)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   pin_memory=True,
                                                   shuffle=False,
                                                   sampler=DistributedSampler(train_dataset))
        return train_loader
    
    def get_val_loader(self, txt):
        val_dataset = KAIST_val(dir = os.path.join(self.config.data.data_dir, self.config.data.dataset), 
                                  transforms=self.transforms,
                                  filelist=txt)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return val_loader
    
class KAIST_train:
    def __init__(self, dir, patch_size, n, transforms, filelist=None):
        super().__init__()

        self.dir = dir
        train_list = os.path.join(dir, filelist)
        print("==> load image list = {}".format(train_list))
        with open(train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() + ".png" for i in contents]
            gt_names = [i.strip().replace('/lwir/', '/visible/') for i in input_names]
            logits_names = [i.strip().replace('/lwir/', '/lwir_seg/').replace(".png", "_logits.npy") for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.logits_names = logits_names
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw
    
    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)
    
    @staticmethod
    def crop_logits(logits, i_list, j_list, h, w):
        crops = []
        for i in range(len(i_list)):
            new_crop = logits[:, i_list[i]:i_list[i]+h, j_list[i]:j_list[i]+w]
            crops.append(new_crop)
        return tuple(crops)
    
    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        logits_name = self.logits_names[index]
        img_id = Path(input_name).stem
        input_img = Image.open(os.path.join(self.dir, input_name))
        input_img = input_img.convert("RGB")
        
        gt_img = Image.open(os.path.join(self.dir, gt_name))

        # load logits
        logits = np.load(os.path.join(self.dir, logits_name))

        i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
        input_img = self.n_random_crops(input_img, i, j, h, w)
        gt_img = self.n_random_crops(gt_img, i, j, h, w)
        logits = self.crop_logits(logits, i, j, h, w)
        outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                    for i in range(self.n)]
        logits_out = [self.transforms(l).permute(1,2,0) for l in logits]
        return torch.stack(outputs, dim=0), img_id, torch.stack(logits_out, dim=0)
    

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
    
class KAIST_val:
    def __init__(self, dir, transforms, filelist=None):
        super().__init__()
        self.dir = dir
        val_list = os.path.join(dir, filelist)
        
        print("==> load image list = {}".format(val_list))
        
        with open(val_list) as f:
            contents = f.readlines()
            input_names = [i.strip() + ".png" for i in contents]
            gt_names = [i.strip().replace('/lwir/', '/visible/') for i in input_names]
            logits_names = [i.strip().replace('/lwir/', '/lwir_seg/').replace(".png", "_logits.npy") for i in input_names]
        
        matches_dir = str(Path(input_names[0]).parent)
        matches_dir = matches_dir.replace('/lwir', '/lwir_matches')
        print(matches_dir)
        matches_names = [None]
        for i in range(1, len(input_names)):
            curr_name = Path(input_names[i]).stem
            prev_name = Path(input_names[i-1]).stem
            matches_names.append(os.path.join(matches_dir, prev_name + "_and_" + curr_name + ".npy"))
        self.matches_names = matches_names
        self.input_names = input_names
        self.gt_names = gt_names
        self.logits_names = logits_names
        self.transforms = transforms

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        logits_name = self.logits_names[index]

        img_id = Path(input_name).stem

        img = Image.open(os.path.join(self.dir, input_name)).convert("RGB")
        img = self.transforms(img)

        gt_img = Image.open(os.path.join(self.dir, gt_name))
        gt_img = self.transforms(gt_img)

        logits = np.load(os.path.join(self.dir, logits_name))
        logits = self.transforms(logits).permute(1,2,0)

        if index == 0:
            matches = torch.empty(0)
        else:
            matches = np.load(os.path.join(self.dir, self.matches_names[index]))
            matches = torch.from_numpy(matches)
        
        return img, logits, gt_img, img_id, matches

    def __getitem__(self, index):
        res = self.get_images(index)
        return res
    
    def __len__(self):
        return len(self.input_names)