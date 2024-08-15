import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet
from torch.nn.parallel import DistributedDataParallel


# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.parallel.DistributedDataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.parallel.DistributedDataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.parallel.DistributedDataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b, logits):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    x_in = torch.cat([x0[:, :3, :, :], logits, x], dim=1)
    output = model(x_in, t.float())
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)


class DenoisingDiffusion(object):
    def __init__(self, args, config, phase):
        super().__init__()
        self.args = args
        self.config = config
        self.device = int(os.environ["LOCAL_RANK"]) if phase == 'train' else config.device

        self.model = DiffusionUNet(config)
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model) if phase == 'train' else self.model
        self.model.to(self.device)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        self.model = DistributedDataParallel(self.model, device_ids=[self.device]) if phase == 'train' else self.model

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader = DATASET.get_train_loader(txt=self.args.txt)

        

        for epoch in range(self.start_epoch, self.config.training.n_iters):
            print('epoch: ', epoch)
            epoch_start = time.time()
            b_sz = len(next(iter(train_loader))[0])
            print(f"[GPU{self.device}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(train_loader)}")
            train_loader.sampler.set_epoch(epoch)
            for i, (x, y, logits) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                logits = logits.flatten(start_dim=0, end_dim=1) if logits.ndim == 5 else logits
                n = x.size(0)
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                
                logits = logits.to(self.device)

                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(self.model, x, t, e, b, logits=logits)

                if self.step % 10 == 0:
                    print(f"step: {self.step}, loss: {loss.item()}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.device == 0 and (self.step % self.config.training.snapshot_freq == 0 or self.step == 1):
                    print(f"[GPU{self.device}] Save checkpoint at step: {self.step}")
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.module.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.work_dir, self.config.data.dataset, 'weights', 'model' + str(self.step).zfill(8)))
                
                if self.step == self.config.training.n_iters:
                    break
            if self.step == self.config.training.n_iters:
                break
            epoch_time = str(round(time.time() - epoch_start, 2))
            print(f"\t===> epoch time = {epoch_time}s")
            

    def sample_image(self, x_cond, x, logits=None, last=True, patch_locs=None, patch_size=None, matches=None, xs_prev=None):
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                              corners=patch_locs, p_size=patch_size,
                                                              logits=logits, matches=matches, xs_prev_frame=xs_prev, 
                                                              decay_factor=self.args.decay_factor)
        else:
            xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0., logits=logits)
        if last:
            xs = xs[-1]
        return xs
    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.config.data.work_dir, self.config.data.dataset, 'results')
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y, logits) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                logits = logits.flatten(start_dim=0, end_dim=1) if logits.ndim == 5 else logits
                break
            n = x.size(0)
            x_cond = x[:, :3, :, :].to(self.device)
            x_cond = data_transform(x_cond)
            logits = logits.to(self.device)
            
            x = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
            x = self.sample_image(x_cond, x, logits=logits)
            x = inverse_data_transform(x)
            x_cond = inverse_data_transform(x_cond)

            for i in range(n):
                x_cond_ = x_cond[i].to("cpu")
                x_save = torch.concat((x_cond_, x[i]), axis=2)
                utils.logging.save_image(x_save, os.path.join(image_folder, str(step), f"{i}.png"))
                