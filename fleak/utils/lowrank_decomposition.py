import copy

import numpy as np
import torch
from scipy.linalg import svd,qr
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter

def low_rank_decomposition(tensor, rank):
    U, s, Vh = svd(tensor.detach().cpu().numpy(), full_matrices=False)
    A = U[:, :rank] @ np.diag(s[:rank])
    return (
        torch.tensor(A, dtype=torch.float32),
        torch.tensor(Vh[:rank, :], dtype=torch.float32)
    )

def low_rank_decomposition_svd(tensor, rank, prune_ratio, if_Conv):
    # importance = torch.abs(tensor) * torch.abs(tensor.grad)
    threshold = _dynamic_threshold(tensor, prune_ratio)
    mask = (torch.abs(tensor) > threshold)
    tensor = tensor * mask
    # return tensor

    ##SVD
    tensor = tensor.detach().cpu().numpy()
    if if_Conv:
        original_shape = tensor.shape
        tensor = tensor.reshape(original_shape[0], -1)
    U, s, Vt = svd(tensor, full_matrices=True)
    # U_1 = copy.deepcopy(U)
    # s_1 = copy.deepcopy(s)
    # Vt_1 = copy.deepcopy(Vt)
    if rank > min(tensor.shape):
        # rank = 5
        rank = _curvature_based_rank_selection(s, 0.95)
        # total_energy = np.sum(s ** 2)
        # cumulative_energy = np.cumsum(s ** 2) / total_energy
        # rank = np.argmax(cumulative_energy >= 0.95) + 1

    U = U[:, :rank]
    Vt = Vt[:rank, :]
    U_ternary, s_u, U_residual = _ternarize(U, 0.3)
    V_ternary, s_v, V_residual = _ternarize(Vt, 0.3)


    return (
        (torch.tensor(U_ternary, dtype=torch.float32), torch.tensor(s_u, dtype=torch.float32)),
        torch.tensor(s[:rank], dtype=torch.float32),
        (torch.tensor(V_ternary, dtype=torch.float32), torch.tensor(s_v, dtype=torch.float32)),
        torch.tensor(U_residual, dtype=torch.float32),
        torch.tensor(V_residual, dtype=torch.float32)
    )

def cnn_prune_terngrad(tensor,prune_ratio, if_Conv):
    # importance = torch.abs(tensor) * torch.abs(tensor.grad)
    original_shape = tensor.shape
    threshold = _dynamic_threshold(tensor, prune_ratio)
    mask = (torch.abs(tensor) > threshold)
    tensor = tensor * mask

    tensor = tensor.detach().cpu().numpy()

    if if_Conv:
        tensor = tensor.reshape(original_shape[0], -1)

    tensor_ternary, s_t = channelwise_ternarize(tensor, 0.3)
    return((torch.tensor(tensor_ternary, dtype=torch.float32), torch.tensor(s_t, dtype=torch.float32)))


def _dynamic_threshold(tensor, prune_ratio):
    abs_values = torch.abs(tensor).flatten()
    k = int(abs_values.numel() * (1 - prune_ratio))
    if k == 0: return 0
    threshold = torch.topk(abs_values, k, largest=True).values[-1]
    return threshold.item()



def _ternarize(array, ratio=0.3):
    abs_matrix = np.abs(array.flatten())
    threshold = np.quantile(abs_matrix, 1 - ratio)
    # threshold = _dynamic_threshold_terngrad(array, ratio)

    ternary = np.zeros_like(array, dtype=np.int8)

    ternary[array > threshold] = 1
    ternary[array < -threshold] = -1

    mask = (ternary != 0)
    if np.any(mask):
        scale = np.sum(np.abs(array) * mask) / (np.sum(mask) + 1e-9)
    else:
        scale = 0.0

    approx = ternary * scale
    residual = array - approx

    return ternary, scale.astype(array.dtype), residual.astype(array.dtype)



def channelwise_ternarize(tensor, threshold_ratio=0.2):
    # tensor shape: [C, ...]
    abs_matrix = np.abs(tensor.flatten())
    threshold = np.quantile(abs_matrix, 1 - threshold_ratio)

    ternarized = np.zeros_like(tensor, dtype=np.int8)
    for i in range(tensor.shape[0]):
        channel = tensor[i]
        th = np.max(np.abs(channel)) * threshold_ratio
        ternarized[i][channel > th] = 1
        ternarized[i][channel < -th] = -1

    mask = (ternarized != 0)
    if np.any(mask):
        scale = np.sum(np.abs(tensor) * mask) / (np.sum(mask) + 1e-9)
    else:
        scale = 0.0
    return ternarized,scale.astype(tensor.dtype)



def _curvature_based_rank_selection(s, energy_threshold=0.95, epsilon=1e-6):
    energy = s ** 2
    total_energy = np.sum(energy)
    cumulative_energy = np.cumsum(energy) / total_energy

    smoothed_energy = savgol_filter(cumulative_energy, window_length=5, polyorder=3)

    second_deriv = np.gradient(np.gradient(smoothed_energy))

    curvature = np.abs(second_deriv) / (1 + np.gradient(smoothed_energy) ** 2) ** 1.5
    knee_point = np.argmax(curvature)

    energy_rank = np.argmax(cumulative_energy >= energy_threshold) + 1
    return min(max(knee_point, energy_rank), len(s))


def low_rank_decomposition_qr_mask(tensor, rank):
    tensor = tensor.detach().cpu().numpy()
    Q, R = qr(tensor)
    mask_scale = 0.01
    ## random mask
    mask_prob = 0.2
    mask_Q = np.random.rand(*Q.shape) > mask_prob
    mask_R = np.random.rand(*R.shape) > mask_prob
    Q = Q * mask_Q
    R = R * mask_R

    return (torch.tensor(Q, dtype=torch.float32),
            torch.tensor(R, dtype=torch.float32))

def low_rank_decomposition_nmf(tensor, rank):
    # param = tensor.detach().cpu().numpy()
    nmf = NMF(n_components=rank, init='random', random_state=42)
    w = nmf.fit_transform(tensor.detach().cpu().numpy())
    H = nmf.components_
    return (
        torch.tensor(w, dtype=torch.float32),
        torch.tensor(H, dtype=torch.float32)
    )

def svd_cnn(tensor,rank):
    ## turn
    flat_weight = tensor.view(tensor.size(0) * tensor.size(1), -1).cpu().numpy()
    U, s, Vt = svd(flat_weight, full_matrices=True)

    return (
        torch.tensor(U[:, :rank], dtype=torch.float32),
        torch.tensor(np.diag(s[:rank]), dtype=torch.float32),
        torch.tensor(Vt[:rank, :], dtype=torch.float32)
    )


def compensate_low_rank(array1,array2,alpha=0.2):
    residual = array1 - array2
    return array2 + alpha * residual


def reconstruct_from_low_rank(U, S, Vt,s_u=None,s_v=None,U_residual=None, V_residual=None):
    if s_u is not None:
        U = U.float() * s_u
    if s_v is not None:
        Vt = Vt.float() * s_v
    if U_residual is not None:
        U += U_residual
    if V_residual is not None:
        Vt += V_residual
    S = torch.diag(S)
    return U @ S @ Vt

def reconstruct_from_lr_cnn(tensor, s_t):
    return tensor.float() * s_t

def reconstruct_from_low_rank_nmf(W,H):
    return torch.matmul(W,H)

def reconstruct_from_QR(Q,R):
    return torch.mm(Q,R)

def check_validity(Q,R):
    if np.isnan(Q).any():
        raise ValueError("Q 包含 NaN")
    if np.isnan(R).any():
        raise ValueError("R 包含 NaN")

