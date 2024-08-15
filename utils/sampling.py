import torch
import utils.logging
import os
import torchvision
from torchvision.transforms.functional import crop



# This script is adapted from the following repository: https://github.com/ermongroup/ddim

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def generalized_steps(x, x_cond, seq, model, b, eta=0., logits=None):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            x_in = torch.cat([x_cond, logits, xt], dim=1)
            et = model(x_in, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs, x0_preds


# def generalized_steps_overlapping(x, x_conds, seq, model, b, eta=0., corners=None, p_size=None, manual_batching=True, logits=None):
#     with torch.no_grad():
#         device = x[0].device
#         n = x[0].size(0)
#         seq_next = [-1] + list(seq[:-1])
#         x0_preds = []
#         xs = [[r] for r in x] # list of list

#         x_grid_masks = [torch.zeros_like(x_cond, device=device) for x_cond in x_conds]
#         for x_grid_mask in x_grid_masks:
#             for (hi, wi) in corners:
#                 x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

#         for k, j in zip(reversed(seq), reversed(seq_next)):
#             t = (torch.ones(n) * k).to(device)
#             next_t = (torch.ones(n) * j).to(device)
#             at = compute_alpha(b, t.long())
#             at_next = compute_alpha(b, next_t.long())
#             xt = [xs_[-1].to(device) for xs_ in xs]
#             et_outputs = [torch.zeros_like(x_cond, device=device) for x_cond in x_conds]
            
#             if manual_batching:
#                 manual_batching_size = 64
#                 xt_patch = [torch.cat([crop(xt_, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0) for xt_ in xt]
#                 x_cond_patch = [torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0) for x_cond in x_conds]
#                 logits_patch = [torch.cat([crop(logits_, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0) for logits_ in logits]
#                 for i in range(0, len(corners), manual_batching_size):
#                     x_in = [torch.cat([x_cond_patch_[i:i+manual_batching_size], 
#                                       logits_patch_[i:i+manual_batching_size], 
#                                       xt_patch_[i:i+manual_batching_size]], dim=1) 
#                                       for (x_cond_patch_, logits_patch_, xt_patch_) in zip(x_cond_patch, logits_patch, xt_patch)]
#                     outputs = [model(x_in_, t) for x_in_ in x_in]
#                     for s_i, output in enumerate(outputs):
#                         for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
#                             et_outputs[s_i][0, :, hi:hi + p_size, wi:wi + p_size] += output[idx]
 

#             et = [torch.div(et_output, x_grid_mask) for (et_output, x_grid_mask) in zip(et_outputs, x_grid_masks)]
#             x0_t = [(xt_ - et_ * (1 - at).sqrt()) / at.sqrt() for (xt_, et_) in zip(xt, et)]
#             # x0_preds.append(x0_t.to('cpu'))

#             c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
#             c2 = ((1 - at_next) - c1 ** 2).sqrt()
#             xt_next = [at_next.sqrt() * x0_t_ + c1 * torch.randn_like(x_) + c2 * et_ for (x0_t_, et_, x_) in zip(x0_t, et, x)] 
#             for s_i, xt_next_ in enumerate(xt_next): 
#                 xs[s_i].append(xt_next_.to('cpu'))
#     return xs



def generalized_steps_overlapping(x, x_cond, seq, model, b, eta=0., corners=None, p_size=None, logits=None, matches=None, xs_prev_frame=None, decay_factor=0.85):
    with torch.no_grad():
        im_hgt, im_wid = x.shape[2], x.shape[3]
        device = x.device
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x] # list of list

        x_grid_mask = torch.zeros_like(x_cond)
        
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        weight = 1
        prev_i = 1
        for k, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * k).to(device)
            next_t = (torch.ones(n) * j).to(device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to(device)

            # xt_next = [torch.zeros((x_cond.shape[0], x_cond.shape[1], x_cond.shape[2]+2*padding, x_cond.shape[3]+2*padding), device=device) 
            #            for x_cond in x_conds]
            xt_next = torch.zeros_like(x_cond)
            
            manual_batching_size = 64
            xt_patch = torch.cat([crop(xt, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
            x_cond_patch = torch.cat([data_transform(crop(x_cond, hi, wi, p_size, p_size)) for (hi, wi) in corners], dim=0)
            logits_patch = torch.cat([crop(logits, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)

            # process patches of each image within chunk
            for i in range(0, len(corners), manual_batching_size):
                x_cond_patch_1 = x_cond_patch[i:i+manual_batching_size]
                xt_patch_1 = xt_patch[i:i+manual_batching_size]
                x_in = torch.cat([x_cond_patch_1, logits_patch[i:i+manual_batching_size], xt_patch_1], dim=1)
                et_patch = model(x_in, t)

                for idx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                    x0_patch_t = (xt_patch_1[idx] - et_patch[idx] * (1 - at).sqrt()) / at.sqrt()
                    c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                    c2 = ((1 - at_next) - c1 ** 2).sqrt()
                    xt_patch_next = at_next.sqrt() * x0_patch_t + c1 * torch.randn_like(x0_patch_t) + c2 * et_patch[idx]

                    # xt_next[img_idx][:, :, hi+padding:hi+padding + p_size, wi+padding:wi+padding + p_size] += xt_patch_next
                    xt_next[:, :, hi:hi + p_size, wi:wi + p_size] += xt_patch_next
            
            xt_next = torch.div(xt_next, x_grid_mask)
            if len(xs_prev_frame) > 0:
                x_prev_frame = xs_prev_frame[prev_i]
                xt_next[:,:,matches[:,3],matches[:,2]] = (1-weight)*xt_next[:,:,matches[:,3],matches[:,2]] + \
                                                            weight*x_prev_frame[:,:,matches[:,1], matches[:,0]]
                weight *= decay_factor
                prev_i += 1
            
            xs.append(xt_next)
    return xs
