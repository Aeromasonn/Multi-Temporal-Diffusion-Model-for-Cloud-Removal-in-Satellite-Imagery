import matplotlib.pyplot as plt
import torch

def edm_preconditioning(sigma, sigma_data=1.0):
    """
    Given noise level sigma (shape [B]), compute EDM-style preconditioning coefficients.

    Returns:
        c_in, c_skip, c_out, c_noise
    each broadcastable to image tensors.
    """
    # make sure sigma has shape [B, 1, 1, 1]
    if sigma.ndim == 1:
        sigma = sigma.view(-1, 1, 1, 1)

    sigma2 = sigma ** 2
    sd2 = sigma_data ** 2

    # input scaling
    c_in = 1.0 / torch.sqrt(sd2 + sigma2)

    # skip connection weight
    c_skip = sd2 / (sd2 + sigma2)

    # output scaling
    c_out = (sigma * sigma_data) / torch.sqrt(sd2 + sigma2)
    c_noise = 0.25 * torch.log(sigma)

    return c_in, c_skip, c_out, c_noise

def visualize(cloudy_seq, batch, x0):
    def to_vis(img):
        # Accepts (4,H,W) or (3,H,W);
        # -> If (T,4,H,W) take mean over time
        if img.ndim == 4:
            img = img.mean(dim=0)
        if img.ndim != 3:
            raise ValueError(f"Expected 3 dims after squeeze, got {img.ndim}")
        x = img[:3].detach().cpu().numpy()
        x = x.transpose(1, 2, 0)  # (H,W,C)
        x = x - x.min()
        x = x / (x.max() + 1e-6)
        return x

    i = 0
    cloudy_vis = to_vis(cloudy_seq[i])
    clean_vis = to_vis(batch["clean"][i])
    x0_vis = to_vis(x0[i])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Cloudy")
    plt.imshow(cloudy_vis)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.title("Target Clean")
    plt.imshow(clean_vis)
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.title("Removal Output")
    plt.imshow(x0_vis)
    plt.axis("off")
    plt.show()

def count_params(module):
    return sum(p.numel() for p in module.parameters())