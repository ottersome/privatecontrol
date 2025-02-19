from typing import Dict, List

import torch
from torch import nn

def calculate_validation_metrics(
    all_features: torch.Tensor,
    pub_features_idxs: List[int],
    prv_features_idxs: List[int],
    model_vae: nn.Module,
    model_adversary: nn.Module,
) -> Dict[str, float]:
    """
    We use correlation here as our delta-epsilon metric.
    """
    prv_features = all_features[:, :, prv_features_idxs]
    pub_features = all_features[:, :, pub_features_idxs]

    # Run data through models.
    latent_z, sanitized_data, kl_divergence = model_vae(all_features[:,:-1,:]) # Do not leak the last element of sequence
    recon_pub = sanitized_data
    recon_priv = model_adversary(latent_z)

    # Lets just do MSE for now
    pub_mse = torch.mean((pub_features[:, -1, :].squeeze() - recon_pub) ** 2)
    prv_mse = torch.mean((prv_features[:, -1, :].squeeze() - recon_priv) ** 2)

    # TODO: Do torch equivalent so we can get the correlation coefficient of the sequences predcicted
    # corr_pub = torch.corrcoef(pub_features[:, -1, :].flatten(), recon_pub.flatten())[0, 1]
    # corr_prv = torch.corrcoef(prv_features[:, -1, :].flatten(), recon_priv.flatten())[0, 1]
    validation_metrics = {
        "pub_mse": pub_mse.item(),
        "prv_mse": prv_mse.item(),
    }

    return validation_metrics
