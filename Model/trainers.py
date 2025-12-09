import torch.nn.functional as F
import torch
from .utils import edm_preconditioning

def forward_trainer(epochs, train_loader, optimizer,
                    forwarder, cloud_encoder, denoiser, device):
    cloud_encoder.train()
    denoiser.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            cloudy_seq = batch['cloudy_seq'].to(device)
            clean      = batch['clean'].to(device)
            B = cloudy_seq.shape[0]

            cloudy_ref = cloudy_seq[:, 0]

            feat_agg, z_agg = cloud_encoder(cloudy_seq)

            feat_up = F.interpolate(
                feat_agg,
                size=cloudy_ref.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )

            cloudy_seq_flat = cloudy_seq.view(B, -1, cloudy_seq.shape[-2], cloudy_seq.shape[-1])

            t = forwarder.sample_t(B, device=device)
            x_t, eps, mu_t = forwarder(clean, cloudy_ref, t)

            sigma = forwarder.sigmas[t].to(device)       # (B,)

            c_in, c_skip, c_out, c_noise = edm_preconditioning(
                sigma, sigma_data=1.0
            )

            x_t_scaled = c_in * x_t

            # F_in: noisy + full cloudy sequence + upsampled features
            F_in = torch.cat([x_t_scaled, cloudy_seq_flat, feat_up], dim=1)  # (B,?,H,W)

            F_out = denoiser(F_in, t, z_agg)

            pred_x0 = c_skip * x_t + c_out * F_out

            lambda_sigma = 1.0 / (c_out ** 2 + 1e-8)
            loss = (lambda_sigma * (pred_x0 - clean) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        print(f"[Epoch {epoch+1}] loss = {avg_loss:.6f}")
