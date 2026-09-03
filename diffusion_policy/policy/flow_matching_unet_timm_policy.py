"""
Flow Matching variant of DiffusionUnetTimmPolicy.

Same obs_encoder, ConditionalUnet1D architecture, dataset, and normalizer as
the diffusion (DDIM) policy - only the training objective and the sampling
procedure differ:
    - Diffusion: network predicts noise eps added at a random diffusion
      timestep (integer 0..num_train_timesteps); sampling denoises with a
      DDIM scheduler.
    - Flow Matching: network predicts the velocity v = x1 - x0 along the
      straight-line path xt = (1-t)*x0 + t*x1 between data x0 and noise x1,
      t in [0, 1]; sampling integrates the ODE with a simple Euler step,
      typically needing far fewer steps.

IMPORTANT - timestep scale: ConditionalUnet1D's SinusoidalPosEmb assumes the
timestep argument spans roughly the same range as the diffusion scheduler's
num_train_timesteps (e.g. 0..50). Flow matching's t in [0, 1] is far too
small a range for that embedding: most of its frequency channels
(emb[k] = exp(-k * const), k up to diffusion_step_embed_dim//2) would be
multiplied by numbers close to 0, collapsing the embedding to a near-constant
vector. `time_scale` rescales t before it reaches the embedding so it spans a
similar effective range to the diffusion case. Must be identical at train and
inference time (it's saved in cfg / the checkpoint's policy config).
"""
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class FlowMatchingUnetTimmPolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            obs_encoder: TimmObsEncoder,
            num_inference_steps: int = 8,
            time_scale: float = 50.0,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            # parameters passed to step (kept for config-compat with the
            # diffusion policy's `policy.kwargs`; unused here since there is
            # no scheduler.step call)
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())

        # create flow-matching velocity model (same architecture as diffusion)
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.num_inference_steps = num_inference_steps
        self.time_scale = float(time_scale)
        self.kwargs = kwargs

    # ========= inference  ============
    def conditional_sample(self,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            **kwargs
        ):
        model = self.model

        # x1 (t=1): pure noise, matching the training convention x1=noise
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        n = self.num_inference_steps
        dt = 1.0 / n

        # Euler-integrate the ODE dx/dt = v_theta(x, t) from t=1 down to t=0.
        for i in range(n):
            t = 1.0 - i * dt

            # 1. apply conditioning (inpaint fixed prefix, if any)
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict velocity at this t
            t_batch = torch.full(
                (trajectory.shape[0],), t,
                device=trajectory.device, dtype=trajectory.dtype)
            v = model(trajectory, t_batch * self.time_scale,
                local_cond=local_cond, global_cond=global_cond)

            # 3. Euler step towards t=0 (towards the data x0)
            trajectory = trajectory - dt * v

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)

        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)

        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple flow-matching samples per obs (same trick as the
        # diffusion policy: repeat the (expensive) obs encoding, cheaply draw
        # a different noise/t pair per repeat)
        if self.train_diffusion_n_samples != 1:
            global_cond = torch.repeat_interleave(global_cond,
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions,
                repeats=self.train_diffusion_n_samples, dim=0)

        x0 = nactions                          # data
        x1 = torch.randn_like(x0)              # noise

        B = x0.shape[0]
        t = torch.rand(B, device=x0.device, dtype=x0.dtype)  # t ~ U(0, 1)
        t_ = t.view(B, 1, 1)

        xt = (1.0 - t_) * x0 + t_ * x1          # point on the straight path
        target_v = x1 - x0                      # velocity along the path (constant in t)

        pred_v = self.model(
            xt,
            t * self.time_scale,
            local_cond=None,
            global_cond=global_cond
        )

        loss = F.mse_loss(pred_v, target_v, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
