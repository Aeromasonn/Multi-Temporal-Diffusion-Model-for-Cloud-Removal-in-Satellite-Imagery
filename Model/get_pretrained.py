import torch
from .model import build_forward_model
from .utils import count_params

def get_pretrained_small(device, cloud_enc_pth, denoiser_pth):
    model_dict = build_forward_model(
    in_channels = 4,
    base_channels = 32,
    num_stages = 2,
    latent_dim = 128,
    T_cloud=3,
    T_diffusion=750,
    device=device
    )

    cloud_encoder = model_dict['cloud_encoder']
    forwarder = model_dict['forwarder']
    denoiser = model_dict['denoiser']

    cloud_encoder.load_state_dict(torch.load(cloud_enc_pth))
    denoiser.load_state_dict(torch.load(denoiser_pth))

    cloud_enc_count = count_params(cloud_encoder)
    forward_enc_count = count_params(forwarder)
    denoiser_count = count_params(denoiser)
    print('Pretrained small model loaded successfully.')
    print(f"The model has {cloud_enc_count + forward_enc_count + denoiser_count} parameters.")

    return cloud_encoder, forwarder, denoiser


def get_pretrained_large(device, cloud_enc_pth, denoiser_pth):
    model_dict = build_forward_model(
    in_channels = 4,
    base_channels = 32,
    num_stages = 3,
    latent_dim = 256,
    T_cloud=3,
    T_diffusion=750,
    device=device
    )
    """
    Models in '../pretrained' take this configuration.
    """

    cloud_encoder = model_dict['cloud_encoder']
    forwarder = model_dict['forwarder']
    denoiser = model_dict['denoiser']

    cloud_encoder.load_state_dict(torch.load(cloud_enc_pth))
    denoiser.load_state_dict(torch.load(denoiser_pth))

    cloud_enc_count = count_params(cloud_encoder)
    forward_enc_count = count_params(forwarder)
    denoiser_count = count_params(denoiser)
    print('Pretrained large model loaded successfully.')
    print(f"The model has {cloud_enc_count+forward_enc_count+denoiser_count} parameters.")

    return cloud_encoder, forwarder, denoiser