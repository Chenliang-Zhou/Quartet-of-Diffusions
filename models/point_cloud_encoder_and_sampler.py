import os
import sys
import torch
from diffusers import DDPMScheduler

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.extend([os.path.join(this_dir, d) for d in ['../..', '../../LION']])
from global_utils import requires_grad
from LION.models.shapelatent_modules import PointNetPlusEncoder
from LION.models.score_sde.resnet import PriorSEDrop
# from LION.models.lion import LION
from LION.models.distributions import Normal
from LION.default_config import cfg as config


# get pretrained point cloud encoder and the DPM for the global prior, ie the DPM for style code, LION
# cate: "airplane", "chair", or "car"
def get_pretrained_encoder_and_sampler(cate):
    model_config = f"/local/scratch/cz363/phd-research/pretrained/LION/{cate}/cfg.yml"
    config.merge_from_file(model_config)

    encoder = PointNetPlusEncoder(zdim=config.latent_pts.style_dim, input_dim=config.ddpm.input_dim, args=config)
    dpm = PriorSEDrop(config.sde, config.latent_pts.style_dim, config)
    encoder.load_state_dict(torch.load(f"/local/scratch/cz363/phd-research/pretrained/LION/{cate}/style_encoder.pt"))
    dpm.load_state_dict(torch.load(f"/local/scratch/cz363/phd-research/pretrained/LION/{cate}/style_prior.pt"))
    requires_grad(encoder, False)
    requires_grad(dpm, False)

    scheduler = scheduler = DDPMScheduler(clip_sample=False,
                                          beta_start=config.ddpm.beta_1, beta_end=config.ddpm.beta_T,
                                          beta_schedule=config.ddpm.sched_mode,
                                          num_train_timesteps=config.ddpm.num_steps,
                                          variance_type=config.ddpm.model_var_type)

    def encode(x):
        z = encoder(x)
        z_mu, z_sigma = z['mu_1d'], z['sigma_1d']  # log_sigma
        dist = Normal(mu=z_mu, log_sigma=z_sigma)  # (B, F)
        return dist.sample()[0]
    encoder.encode = encode

    def sample(num_samples):
        device = next(dpm.parameters()).device
        scheduler.set_timesteps(1000, device=device)
        timesteps = scheduler.timesteps
        assert not dpm.mixed_prediction
        sampled_list = []

        # start sampling
        x_T_shape = [num_samples, config.latent_pts.style_dim, 1, 1]
        x_noisy = torch.randn(size=x_T_shape, device=device)
        condition_input = None
        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device=device) * (t + 1)
            noise_pred = dpm(x=x_noisy, t=t_tensor.float(), condition_input=condition_input, clip_feat=False)
            x_noisy = scheduler.step(noise_pred, t, x_noisy).prev_sample
        sampled_list.append(x_noisy)
        return torch.cat(sampled_list, dim=1).squeeze(2).squeeze(2)
    dpm.sample = sample

    return encoder, dpm


if __name__ == "__main__":
    cate = "airplane"
    encoder, dpm = get_pretrained_encoder_and_sampler(cate)
    print(dpm)
    pc = torch.randn(5, 2048, 3).cuda()
    latent = encoder.cuda().encode(pc)
    print("Latent shape:", latent.shape)
    samples = dpm.cuda().sample(5)
    print("Sample shape:", samples.shape)
    print("Done")
