import torch
from torch import nn
from statistics import mean
import numpy as np
import torch.nn.functional as F

def validate_vae_recon(
    model_vae: nn.Module,
    all_valid_seqs: torch.Tensor, # X
    validation_batch_size: int,
    pub_columns: list[int],
) -> torch.Tensor:
    """
    Will return 
    Args: 
        original_timeseries: The truth value of this training
        inferred_timeseries: The inferred (by vae model) values of truth value.
    Returns: 
        validation_metric: MSE between the validation and inferred timeseries;
    """

    num_batch_elements = all_valid_seqs.shape[0]

    model_vae.eval()
    validation_losses = []
    with torch.no_grad():
        batch_offsets = np.arange(0, num_batch_elements, validation_batch_size)
        for batch_offset in batch_offsets:
            cur_batch_end = min(batch_offset + validation_batch_size, num_batch_elements)
            batch_all = all_valid_seqs[batch_offset: cur_batch_end, :, :]
            batch_pub = all_valid_seqs[batch_offset: cur_batch_end, : , pub_columns]

            # Then we copy this and shot it to the other ones
            # print(f"Size of `batch_puv`: {batch_pub.shape}")
            # print(f"Size of `batch_all`: {batch_all.shape}")
            _, reconstruted_data, _ = model_vae(batch_all[:,:-1,:])
            # print(f"Size of `reconstruted_data`: {reconstruted_data.shape}")
            # Evaluate
            validation_loss = F.mse_loss(batch_pub[:,-1,:], reconstruted_data)
            validation_loss_scalar = validation_loss.mean()

            validation_losses.append(validation_loss_scalar.item())
            
    model_vae.train()
    return mean(validation_losses)
