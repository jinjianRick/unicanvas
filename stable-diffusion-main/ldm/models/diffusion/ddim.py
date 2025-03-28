"""SAMPLING ONLY."""

import torch, pickle
import numpy as np
from tqdm import tqdm
from functools import partial

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        if type(model)==list:
            self.model_ldm = model[0]
            self.model = model[1]
        else:
            self.model_ldm = model
            self.model = model
        self.model_ldm = model
        self.ddpm_num_timesteps = self.model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               conditioning_ldm=None,
               extra_info=None,
               layout_info=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None,
               # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               **kwargs
               ):
        
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        if conditioning_ldm is not None:
            if isinstance(conditioning_ldm, dict):
                cbs = conditioning_ldm[list(conditioning_ldm.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning_ldm.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning_ldm.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    conditioning_ldm=conditioning_ldm,
                                                    extra_info=extra_info,
                                                    layout_info=layout_info,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      conditioning_ldm=None,
                      extra_info=None,
                      layout_info=None,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,):
        device = self.model.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
            img_ldm = torch.randn(shape, device=device)
        else:
            img = x_T
            img_ldm = torch.randn(img.shape, device=device)

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        intermediates_ldm = {'x_inter': [img_ldm], 'pred_x0': [img_ldm]}

        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        if extra_info is not None and extra_info['b_edit']:
            if extra_info['bk_opt'] == 'get':
                bk_attn={}
            elif extra_info['bk_opt'] == 'use':
                with open(extra_info['bl_path'], 'rb') as f:
                    bk_attn = pickle.load(f)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            
            if extra_info is not None and extra_info['b_edit']:
                if extra_info['bk_opt'] == 'get':
                    extra_info['bk_attn'] = {}
                elif extra_info['bk_opt'] == 'use':
                    extra_info['bk_attn'] = bk_attn[str(ts[0].item())]

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img
                img_ldm = img_orig * mask + (1. - mask) * img_ldm
            
            img_extra = {}
            img_extra['img_ldm'] = img_ldm

            outs = self.p_sample_ddim(img, img_extra, cond,  ts, index=index, cond_ldm=conditioning_ldm, extra_info=extra_info, layout_info=layout_info, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning)

            img, pred_x0, img_ldm, pred_x0_ldm = outs
            img_extra['img_ldm'] = img_ldm
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

                intermediates_ldm['x_inter'].append(img_ldm)
                intermediates_ldm['pred_x0'].append(pred_x0_ldm)
            if extra_info is not None and extra_info['b_edit'] and extra_info['bk_opt'] == 'get':
                bk_attn[str(ts[0].item())] = extra_info['bk_attn']
        
        if extra_info is not None and extra_info['b_edit'] and extra_info['bk_opt'] == 'get':
            with open(extra_info['bl_path'], 'wb') as f:
                pickle.dump(bk_attn, f)

        return img, intermediates

    @torch.no_grad()
    #x_ldm
    def p_sample_ddim(self, x, x_extra, c, t, index, cond_ldm=None, extra_info=None, layout_info=None, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device

        if layout_info is not None:
            layout_info['timesteps'] = t

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c, layout_info=layout_info)
        else:
            if extra_info['b_edit']:
                info_dict_custom = {"type": 'custom', "mask_ldm":extra_info['mask_ldm'], "mask_custom":extra_info['mask_custom'],  "length_ldm":extra_info['length_ldm'], "length_custom":extra_info['length_custom'], "sep_custom":False, "lambda": extra_info['lambda'], 't_tau':extra_info['t_tau'], 'bk_opt':extra_info['bk_opt'], 'bk_attn':extra_info['bk_attn'], 'cor_dict':extra_info['cor_dict']}
            else:
                info_dict_custom = {"type": 'custom', "mask_ldm":extra_info['mask_ldm'], "mask_custom":extra_info['mask_custom'],  "length_ldm":extra_info['length_ldm'], "length_custom":extra_info['length_custom'], "sep_custom":False, "lambda": extra_info['lambda'], 't_tau':extra_info['t_tau']}

            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            x_in_ldm = torch.cat([x_extra['img_ldm']] * 2)
            t_in_ldm = torch.cat([t] * 2)
            x_in = torch.cat([x_in, x_in_ldm], dim=0)
            t_in = torch.cat([t_in, t_in_ldm], dim=0)
            c_in = torch.cat([unconditional_conditioning, c, unconditional_conditioning, cond_ldm['ldm']])
            tmp_e_t = self.model.apply_model(x_in, t_in, c_in, info_dict=info_dict_custom, layout_info=layout_info)
            e_t_uncond, e_t, e_t_uncond_ldm, e_t_ldm = tmp_e_t.chunk(4)
            
            e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            e_t_ldm = e_t_uncond_ldm + unconditional_guidance_scale * (e_t_ldm - e_t_uncond_ldm)
         
        if score_corrector is not None:
            assert self.model.parameterization == "eps"
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
            e_t_ldm = score_corrector.modify_score(self.model, e_t_ldm, x_extra['img_ldm'], t, cond_ldm['ldm'], **corrector_kwargs) ###

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas

        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        pred_x0_ldm = (x_extra['img_ldm'] - sqrt_one_minus_at * e_t_ldm) / a_t.sqrt()

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            pred_x0_ldm, _, *_ = self.model.first_stage_model.quantize(pred_x0_ldm)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        dir_xt_ldm = (1. - a_prev - sigma_t**2).sqrt() * e_t_ldm

        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        noise_ldm = sigma_t * noise_like(x_extra['img_ldm'].shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
            noise_ldm = torch.nn.functional.dropout(noise_ldm, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        x_prev_ldm = a_prev.sqrt() * pred_x0_ldm + dir_xt_ldm + noise_ldm

        return x_prev, pred_x0, x_prev_ldm, pred_x0_ldm

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
        return x_dec