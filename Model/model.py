# -----------------------------
# This python file contains all core models,
# Including: Encoder; Forwarder; Denoiser; Sampler.
# -----------------------------

import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import edm_preconditioning

# ----------------------------
# Model Initialization
# ----------------------------
def build_forward_model(
    device='cpu',
    in_channels=4,
    base_channels=32,
    num_stages=2,
    latent_dim=128,
    T_cloud=3,
    T_diffusion=750,
):
    """
    Build base_encoder, cloud_encoder, forwarder, denoiser and move to device.
    Returns a dict of modules.

    By default a smaller model configuration.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_encoder = CloudEncoder(
        in_channels=in_channels,
        base_channels=base_channels,
        num_stages=num_stages,
        latent_dim=latent_dim,
    ).to(device)

    cloud_encoder = MultiTemporalCloudEncoder(
        base_encoder, T=T_cloud
    ).to(device)

    forwarder = ForwardDiffusion(T=T_diffusion).to(device)

    feat_channels = base_channels * (2 ** (num_stages - 1))
    f_in_channels = 4 + 3 * in_channels + feat_channels

    denoiser = CloudConditionedUNet_4C(
        in_channels=f_in_channels,
        out_channels=4,
        latent_dim=latent_dim,
        max_T=T_diffusion,
    ).to(device)

    return {
        "base_encoder": base_encoder,
        "cloud_encoder": cloud_encoder,
        "forwarder": forwarder,
        "denoiser": denoiser,
        "device": device,
    }

def backward_sampler(cloudy_seq, cloud_encoder, denoiser, forwarder, num_steps):
    """
    Do backward diffusion on cloudy_seq.
    Returns:
        Cloud-removed x0 of input cloudy image.
    """
    x0 = sampler(
        cloudy_seq=cloudy_seq,
        cloud_encoder=cloud_encoder,
        denoiser=denoiser,
        forwarder=forwarder,
        num_steps=num_steps,
    )
    return x0

# ----------------------------
# Cloud Encoders
# ----------------------------
class CloudEncoder(nn.Module):
    """
    CNN cloud encoder.

    Input:
        x: (B, in_channels, H, W), normalize required (should be done by default)

    Output:
        feat: (B, C_lowres, H_down, W_down)  spatial cloud features
        z:    (B, latent_dim)                a global embedding
    """
    def __init__(
        self,
        in_channels = 4,
        base_channels = 32,
        num_stages = 3,
        latent_dim = 128,
        num_groups = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.latent_dim = latent_dim

        layers = []

        # Initial conv to get to base_channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups=min(num_groups, base_channels), num_channels=base_channels),
                nn.SiLU(inplace=True),
            )
        )

        in_ch = base_channels
        channels = [base_channels * (2 ** i) for i in range(num_stages)]

        # Residual + downsample
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for out_ch in channels:
            self.down_blocks.append(ResidualBlock(in_ch, out_ch, num_groups=num_groups))
            # stride-2 conv for downsampling
            self.downsamples.append(
                nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1)
            )
            in_ch = out_ch

        # Residual block at lowest resolution
        self.final_block = ResidualBlock(in_ch, in_ch, num_groups=num_groups)

        # Register first stem as single module for clarity
        self.stem = layers[0]

        # Projection to latent space
        self.proj = nn.Linear(in_ch, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.stem(x)  # (B, base_channels, H, W)
        # Downsampling stages
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h)
            h = down(h)  # spatial size halves each time

        # Final block
        h = self.final_block(h)

        # 1: spatial feature map
        feat = h  # (B, C_final, H_down, W_down)

        # 2: global average pooling -> vector
        pooled = F.adaptive_avg_pool2d(h, output_size=1).squeeze(-1).squeeze(-1)

        # 3: project to latent_dim
        z = self.proj(pooled)  # (B, latent_dim)

        return feat, z

class ResidualBlock(nn.Module):
    """
    Small residual conv block:
    in -> Conv -> GN -> SiLU -> Conv -> GN -> +skip
    """
    def __init__(self, in_channels, out_channels, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(num_groups=min(num_groups, out_channels), num_channels=out_channels)
        self.silu   = nn.SiLU(inplace=True)

        # if channel dims change, use 1*1 conv for skip
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.gn2(out)
        out = out + identity
        out = self.silu(out)
        return out

class MultiTemporalCloudEncoder(nn.Module):
    """
    Wraps a single-image CloudEncoder to handle a sequence of T cloudy images with
    a learned temporal fusion to reduce hallucination risk.
    This takes advantage of 3 cloudy v.s. 1 clean in Sen2-MTC.

    Input:
        cloudy_seq: (B, T, C, H, W)
    Output:
        feat_agg: (B, C_deep, H_d, W_d)   # aggregated spatial features
        z_agg:    (B, latent_dim)         # aggregated global vector
    """
    def __init__(self, base_encoder, T = 3): # 3 cloudy available
        super().__init__()
        self.base_encoder = base_encoder
        self.T = T

        lat_dim = getattr(base_encoder, 'latent_dim', 128)
        hidden = max(lat_dim // 2, 32)
        self.score_mlp = nn.Sequential(
            nn.Linear(lat_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, cloudy_seq: torch.Tensor):
        B, T, C, H, W = cloudy_seq.shape
        assert T == self.T, f"Expected T={self.T}, got {T}"

        x_flat = cloudy_seq.view(B * T, C, H, W)

        feat_flat, z_flat = self.base_encoder(x_flat)
        C_deep = feat_flat.shape[1]
        H_d    = feat_flat.shape[2]
        W_d    = feat_flat.shape[3]
        latent_dim = z_flat.shape[1]

        feat = feat_flat.view(B, T, C_deep, H_d, W_d)
        z    = z_flat.view(B, T, latent_dim)

        # learned temporal weights from z
        scores = self.score_mlp(z)                 # (B,T,1)
        weights = F.softmax(scores, dim=1)         # (B,T,1)

        # weighted aggregation
        w_feat = weights.view(B, T, 1, 1, 1)
        feat_agg = (w_feat * feat).sum(dim=1)      # (B, C_deep, H_d, W_d)
        w_z = weights                               # (B,T,1)
        z_agg = (w_z * z).sum(dim=1)               # (B, latent_dim)

        return feat_agg, z_agg


# ----------------------------
# UNet & Forward Diffusion
# ----------------------------
class ForwardDiffusion(nn.Module):
    r"""
    Forward Diffusion from Liu et al. 2025 (EMRDM):
        see https://arxiv.org/abs/2503.23717

        x_t = (1 - \lambda_t) * clean + \lambda_t * cloudy + \sigma_t * \epsilon
        -- Progressively transform the initial latent state into a noisy version of the 'cloudy' image.
    """

    def __init__(self, T=500, sigma_min=0.01, sigma_max=0.4):
        super().__init__()
        self.T = T

        sigmas = torch.linspace(sigma_min, sigma_max, T)
        lambdas = torch.linspace(0.0, 1.0, T)

        self.register_buffer("sigmas", sigmas)     # (T,)
        self.register_buffer("lambdas", lambdas)   # (T,)

    def sample_t(self, batch_size, device):
        return torch.randint(0, self.T, (batch_size,), device=device)

    def forward(self, clean, cloudy, t):
        """
        clean, cloudy: (B,C,H,W)
        t: (B,): int, timesteps
        Returns:
            x_t:  noisy sample
            eps:  noise used
            mu_t: clean/cloudy mean BEFORE noise
        """
        B = clean.shape[0]
        eps = torch.randn_like(clean)

        sigma_t  = self.sigmas[t].view(B, 1, 1, 1)
        lambda_t = self.lambdas[t].view(B, 1, 1, 1)

        mu_t = (1.0 - lambda_t) * clean + lambda_t * cloudy
        x_t  = mu_t + sigma_t * eps

        return x_t, eps, mu_t

class CloudConditionedUNet_4C(nn.Module):
    def __init__(self, in_channels=4, out_channels=4, latent_dim=128, max_T=750):
        super().__init__()

        self.unet = UNet4C(
            in_channels=in_channels,
            out_channels=out_channels,
            cond_dim=latent_dim,
            max_T=max_T
        )

    def forward(self, x_t, t, z_cloud):
        eps_pred = self.unet(x_t, t, z_cloud)
        return eps_pred

class UNet4C(nn.Module):
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        base_channels=64,
        time_emb_dim=256,
        cond_dim=128,
        max_T=750
    ):
        super().__init__()
        self.max_T = max_T
        # time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # cloud embedding MLP
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ---------- Down ----------
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        self.down1 = ResBlock(base_channels, base_channels * 2, time_emb_dim, time_emb_dim)
        self.down2 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # ---------- Middle ----------
        self.mid = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim, time_emb_dim)

        # ---------- Up ----------
        self.up1 = ResBlock(base_channels * 4 + base_channels * 4, base_channels * 2, time_emb_dim, time_emb_dim)
        self.up2 = ResBlock(base_channels * 2 + base_channels * 2, base_channels, time_emb_dim, time_emb_dim)

        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)

    def forward(self, x, t, z_cloud):
        # t: (B,)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        t = t.float() / (self.max_T - 1)     # normalize to [0,1]
        t_emb = timestep_embedding(t, self.time_mlp[0].in_features)
        t_emb = self.time_mlp(t_emb)

        c_emb = self.cond_mlp(z_cloud)

        # Down
        x1 = self.conv_in(x)
        x2 = self.down1(x1, t_emb, c_emb)
        x3 = self.pool(x2)
        x3 = self.down2(x3, t_emb, c_emb)

        # Middle
        xm = self.mid(x3, t_emb, c_emb)

        # Up
        u1 = torch.cat([xm, x3], dim=1)
        u1 = self.up1(u1, t_emb, c_emb)

        u2 = F.interpolate(u1, scale_factor=2, mode="nearest")
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.up2(u2, t_emb, c_emb)

        out = self.conv_out(u2)
        return out

def timestep_embedding(t, dim):
    """
    Sinusoidal timestep embedding
    """
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, dtype=torch.float32, device=t.device)
        * (-torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb


class ResBlock(nn.Module):
    """
    ResNet block
    Conv -> Norm -> Act -> Conv -> Norm -> Act
    """
    def __init__(self, in_ch, out_ch, time_emb_dim, cond_dim):
        super().__init__()
        self.time_dense = nn.Linear(time_emb_dim, out_ch)
        self.cond_dense = nn.Linear(cond_dim, out_ch)

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()

        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, c_emb):
        """
        x:     (B, C, H, W)
        t_emb: (B, time_emb_dim)
        c_emb: (B, cond_dim)
        """
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        t_added = self.time_dense(t_emb)[:, :, None, None]
        c_added = self.cond_dense(c_emb)[:, :, None, None]
        h = h + t_added + c_added

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + self.shortcut(x)

# ----------------------------
# Sampling & Backward Diffusion
# ----------------------------
@torch.no_grad()
def sampler(
    cloudy_seq,
    cloud_encoder,
    denoiser,
    forwarder,
    num_steps=100,
    device=None,
):
    if device is None:
        device = cloudy_seq.device
    cloudy_seq = cloudy_seq.to(device)   # (B,3,4,H,W)
    B, T, C, H, W = cloudy_seq.shape

    cloudy_ref = cloudy_seq[:, 0] # use first frame
    cloud_feat, z_cloud = cloud_encoder(cloudy_seq)
    feat_up = F.interpolate(
        cloud_feat,
        size=(H, W),
        mode='bilinear',
        align_corners=False,
    )
    cloudy_seq_flat = cloudy_seq.view(B, -1, H, W)         # (B,12,H,W)

    times = torch.linspace(forwarder.T - 1, 0, steps=num_steps + 1, device=device).long()

    t_start = times[0]
    sigma_start = forwarder.sigmas[t_start].view(1, 1, 1, 1).to(device)
    x_t = cloudy_ref + sigma_start * torch.randn_like(cloudy_ref)

    for i in range(num_steps):
        t_now = times[i]
        t_next = times[i + 1]

        lambda_now = forwarder.lambdas[t_now].view(1, 1, 1, 1).to(device)
        lambda_next = forwarder.lambdas[t_next].view(1, 1, 1, 1).to(device)

        sigma_now = forwarder.sigmas[t_now].to(device)
        sigma_next = forwarder.sigmas[t_next].to(device)

        c_in, c_skip, c_out, c_noise = edm_preconditioning(
            sigma_now.expand(B), sigma_data=1.0
        )

        x_t_scaled = c_in * x_t
        F_in = torch.cat([x_t_scaled, cloudy_seq_flat, feat_up], dim=1)  # (B,?,H,W)

        t_batch = t_now.expand(B)
        F_out = denoiser(F_in, t_batch, z_cloud)        # (B,4,H,W)

        x0_pred = c_skip * x_t + c_out * F_out          # (B,4,H,W)

        mu_now  = (1.0 - lambda_now)  * x0_pred + lambda_now  * cloudy_ref
        mu_next = (1.0 - lambda_next) * x0_pred + lambda_next * cloudy_ref

        deviation = x_t - mu_now
        noise_scale = sigma_next / (sigma_now + 1e-8)

        x_t = mu_next + noise_scale * deviation

    return x_t

