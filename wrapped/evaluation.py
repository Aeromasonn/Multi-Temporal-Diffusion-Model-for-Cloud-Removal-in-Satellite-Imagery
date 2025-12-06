import torch
from tqdm import tqdm
import torch.nn.functional as F
from .model import backward_sampler

# ---------- MAE ----------
def mae(pred, target):
    """
    Mean-Squared Error (MAE)
    MSE = 1/N * SUM (x_hat - x)^2
    Lower better.
    """
    return torch.mean(torch.abs(pred - target))

# ---------- PSNR ----------
def psnr(pred, target, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio (PSNR)
    PSNR = 10 * log_10 * (Max_val^2 / MSE)
    Higher better.
    """
    mse = torch.mean((pred - target) ** 2)
    eps = 1e-10
    return 10.0 * torch.log10((max_val ** 2) / (mse + eps))

# ---------- SSIM ----------
def _gaussian_kernel(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    g2d = g[:, None] * g[None, :]     # (W,W)
    g2d = g2d.unsqueeze(0).unsqueeze(0)  # (1,1,W,W)
    g2d = g2d.repeat(channels, 1, 1, 1)  # (C,1,W,W)
    return g2d

def ssim(pred, target, max_val=1.0, window_size=11, sigma=1.5):
    """
    Structural Similarity Index Measure (SSIM)
    SSIM(x, y) = [(2 * μ_x * μ_y + C1) * (2 * σ_xy + C2)] /
                 [(μ_x^2 + μ_y^2 + C1) * (σ_x^2 + σ_y^2 + C2)]
        where μ_x, μ_y are local means,
              σ_x^2, σ_y^2 are local variances,
              σ_xy is local covariance.
    Measures perceptual similarity in terms of luminance, contrast, and structure.
    Range: [-1, 1] in theory, typically [0, 1] in practice.
    Higher better
    """
    B, C, H, W = pred.shape
    device = pred.device
    window = _gaussian_kernel(window_size, sigma, channels=C, device=device)

    mu_x = F.conv2d(pred, window, padding=window_size//2, groups=C)
    mu_y = F.conv2d(target, window, padding=window_size//2, groups=C)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(pred * pred, window, padding=window_size//2, groups=C) - mu_x2
    sigma_y2 = F.conv2d(target * target, window, padding=window_size//2, groups=C) - mu_y2
    sigma_xy = F.conv2d(pred * target, window, padding=window_size//2, groups=C) - mu_xy

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2))
    return ssim_map.mean()

# ---------- LPIPS ----------

def lpips_distance(pred, target, lpips_model=None):
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    LPIPS(x, y) = Σ_l (1 / (H_l * W_l)) * Σ_{c,h,w} w_{l,c} *
                  ( φ_l(x)_{c,h,w} - φ_l(y)_{c,h,w} )^2
        where φ_l are deep features from a pretrained CNN,
              w_{l,c} are learned channel weights.
    Measures perceptual distance using deep feature differences.
    Range: ≥ 0, often roughly [0, 1] for natural images (depends on model).
    Lower better
    """
    if lpips_model is None:
        return None
    # LPIPS usually wants [-1,1] range and 3 channels
    x = pred[:, :3].clamp(0, 1) * 2 - 1
    y = target[:, :3].clamp(0, 1) * 2 - 1
    with torch.no_grad():
        d = lpips_model(x, y)  # shape (B,1) for official impl
    return d.mean()

# ----------------------------
# Evaluation Process
# ----------------------------
def evaluate_batch(x0, batch, max_val=1.0, lpips_model=None, device='cuda'):
    pred   = x0.to(device)
    target = batch["clean"].to(device)

    # An optional clamp
    pred   = pred.clamp(0, max_val)
    target = target.clamp(0, max_val)

    metrics = {
        "MAE": mae(pred, target).item(),
        "PSNR": psnr(pred, target, max_val=max_val).item(),
        "SSIM": ssim(pred, target, max_val=max_val).item()
    }

    lp = lpips_distance(pred, target, lpips_model)
    if lp is not None:
        metrics["LPIPS"] = lp.item()

    return metrics

@torch.no_grad()
def evaluate_over_loader(
    test_loader,
    cloud_encoder,
    denoiser,
    forwarder,
    num_steps=750,
    max_val=1.0,
    lpips_model=None,
    device='cuda',
):
    cloud_encoder.eval()
    denoiser.eval()
    forwarder.eval()

    all_metrics = []  # list of dicts, one per batch

    for batch in tqdm(test_loader):
        cloudy_seq = batch["cloudy_seq"].to(device)

        # --- compute x0 ---
        x0 = backward_sampler(
            cloudy_seq=cloudy_seq,
            cloud_encoder=cloud_encoder,
            denoiser=denoiser,
            forwarder=forwarder,
            num_steps=num_steps,
        )

        # --- evaluate batch with x0 ---
        m = evaluate_batch(
            x0=x0,
            batch=batch,
            max_val=max_val,
            lpips_model=lpips_model,
            device=device,
        )
        all_metrics.append(m)
        #break # un-comment this to run on ONE sample

    # ---------- aggregate statistics ----------
    keys = all_metrics[0].keys()
    summary = {}
    for k in keys:
        vals = torch.tensor([m[k] for m in all_metrics], dtype=torch.float32)
        summary[k + "_mean"] = vals.mean().item()
        summary[k + "_std"]  = vals.std(unbiased=True).item()  # sample std

    return all_metrics, summary
