import numpy as np
import torch
from collections import OrderedDict

def add_dp_noise(global_params, client_params, delta, epsilon, delta_dp):
    param_diff = {}
    for key in client_params:
        param_diff[key] = client_params[key] - global_params[key]

    # total_norm = 0.0
    total_norm = torch.sqrt(sum(torch.sum(v ** 2) for v in param_diff.values())).item()
    safe_norm = max(total_norm, 1e-6)

    # for key in param_diff:
    #     total_norm += torch.sum(param_diff[key] ** 2)
    # total_norm = torch.sqrt(total_norm).item()


    scale = min(1.0, delta / safe_norm)

    clipped_diff = {k: v * scale for k, v in param_diff.items()}

    sigma = (delta * np.sqrt(2 * np.log(1.25 / delta_dp))) / epsilon
    sigma *= 0.5

    noisy_diff = {}
    for key in clipped_diff:
        dim = np.prod(clipped_diff[key].shape)
        scaled_sigma = sigma / np.sqrt(dim)

        noise = torch.randn_like(clipped_diff[key]) * sigma
        noise = torch.clamp(noise, min=-3 * scaled_sigma, max=3 * scaled_sigma)
        noisy_diff[key] = clipped_diff[key] + noise

        if torch.isnan(noisy_diff[key]).any():
            raise ValueError(f"NaN detected in {key} after adding noise")

    noisy_params = {}
    for key in global_params:
        noisy_params[key] = global_params[key] + noisy_diff[key]

    noisy_params_new = OrderedDict(noisy_params.items())
    return noisy_params_new
