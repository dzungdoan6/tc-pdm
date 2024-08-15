import torch
import torch.nn as nn
import utils
import torchvision
import os


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            print("==> resuming model {}".format(args.resume))
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, temp_blend=False, r=None):
        image_folder = self.args.resdir
        print("===> save results in: {}".format(image_folder))
        with torch.no_grad():
            xs_prev = []
            for i, (x_cond, logits, gt, id, m) in enumerate(val_loader):
                print(f"starting processing image {i+1}/{len(val_loader)}: {id[0]}")
                x_cond = x_cond.to(self.diffusion.device)
                logits = logits.to(self.diffusion.device)
                # mx = [[m.to(self.diffusion.device) for m in mx_] for mx_ in mx]
                # mx = [[m.to(self.diffusion.device) for m in my_] for my_ in my]
                m = m.squeeze().to(self.diffusion.device)
                
                if temp_blend:
                    xs_prev = self.diffusive_restoration(x_cond, r=r, logits=logits, m=m, xs_prev=xs_prev)
                else:
                    xs_prev = self.diffusive_restoration(x_cond, r=r, logits=logits, m=m)
                x_output = xs_prev[-1].to("cpu")
                x_output = inverse_data_transform(x_output)

                # save
                x_cond = x_cond.to("cpu")
                x_save = torch.concat((x_cond, x_output), axis=3)
                utils.logging.save_image(x_save, os.path.join(image_folder, f"{id[0]}.png"))

    def diffusive_restoration(self, x_cond, r=None, logits=None, m=None, xs_prev=[]):
        p_size = self.config.data.image_size
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        xs = torch.randn(x_cond.size(), device=self.diffusion.device)
        x_outputs = self.diffusion.sample_image(x_cond, xs, patch_locs=corners, patch_size=p_size, logits=logits, matches=m, last=False, xs_prev=xs_prev)
        return x_outputs

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        return h_list, w_list
