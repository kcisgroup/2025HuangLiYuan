import numpy as np
import torch
from collections import OrderedDict

def add_dp_noise(global_params, client_params, delta, epsilon, delta_dp):
    """
    为客户端模型参数添加差分隐私高斯噪声

    参数:
        server_params: 服务器下发的原始参数字典
        client_params: 客户端训练后的参数字典
        delta (float): L2范数裁剪阈值（敏感度）
        epsilon (float): 隐私预算ε
        delta_dp (float): 松弛项δ（通常小于1e-5）

    返回:
        noisy_params: 添加噪声后的参数字典
    """
    # 计算参数更新量
    param_diff = {}
    for key in client_params:
        param_diff[key] = client_params[key] - global_params[key]

    # 计算更新量的L2范数
    # total_norm = 0.0
    total_norm = torch.sqrt(sum(torch.sum(v ** 2) for v in param_diff.values())).item()
    safe_norm = max(total_norm, 1e-6)  # 防止除以零

    # for key in param_diff:
    #     total_norm += torch.sum(param_diff[key] ** 2)
    # total_norm = torch.sqrt(total_norm).item()


    # 进行L2范数裁剪
    scale = min(1.0, delta / safe_norm)

    clipped_diff = {k: v * scale for k, v in param_diff.items()}

    # 计算高斯噪声标准差
    sigma = (delta * np.sqrt(2 * np.log(1.25 / delta_dp))) / epsilon
    sigma *= 0.5

    # 生成并添加高斯噪声
    noisy_diff = {}
    for key in clipped_diff:
        dim = np.prod(clipped_diff[key].shape)
        scaled_sigma = sigma / np.sqrt(dim)

        noise = torch.randn_like(clipped_diff[key]) * sigma
        noise = torch.clamp(noise, min=-3 * scaled_sigma, max=3 * scaled_sigma)
        noisy_diff[key] = clipped_diff[key] + noise

        if torch.isnan(noisy_diff[key]).any():
            raise ValueError(f"NaN detected in {key} after adding noise")

    # 生成最终上传参数
    noisy_params = {}
    for key in global_params:
        noisy_params[key] = global_params[key] + noisy_diff[key]

    noisy_params_new = OrderedDict(noisy_params.items())
    return noisy_params_new
