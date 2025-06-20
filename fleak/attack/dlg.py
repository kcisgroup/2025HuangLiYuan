"""

DLG based approaches are very sensitive to dropout, BN layers and so on !

"""

import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def dummy_criterion(dummy_pred, dummy_label):
    dummy_onehot_label = F.softmax(dummy_label, dim=-1)
    return torch.mean(torch.sum(- dummy_onehot_label * F.log_softmax(dummy_pred, dim=-1), 1))


def dlg(model, gt_grads, dummy, gt_x, rec_epochs=300, rec_lr=1.0, device="cpu"):
    """ Deep Leakage Gradient

    https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf

    :param model: inferred model
    :param gt_grads: gradients of the ground truth data
    :param dummy: TorchDummy object
    :param rec_epochs: reconstruct epochs
    :param rec_lr: reconstruct learning rate
    :param device: cpu or cuda
    :return: dummy data, dummy label (int)
    """
    # be careful about the influence on model.train() & model.eval()
    # especially dropout layer and bn layer are included in model

    dummy_data = dummy.generate_dummy_input(device)
    dummy_label = dummy.generate_dummy_label(device)
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=rec_lr)  # default lr=1.0
    criterion = dummy_criterion

    pbar = tqdm(range(rec_epochs),
                total=rec_epochs,
                desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
    for _ in pbar:
        def closure():
            optimizer.zero_grad()
            model.zero_grad()  # not necessary

            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, dummy_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, gt_grads):
                grad_diff += ((dummy_g - origin_g)**2).sum()
            grad_diff.backward()

            return grad_diff

        loss = optimizer.step(closure)
        pbar.set_description("Loss {:.6}".format(loss))

    # save the dummy data
    dummy.append(dummy_data)
    # convert dummy label to integer
    rec_dummy_label = torch.argmax(dummy_label, dim=-1)
    # save the dummy label
    dummy.append_label(rec_dummy_label)

    mse = torch.mean((dummy_data - gt_x) ** 2)
    psnr = 10 * torch.log10(1.0 ** 2 / (mse + 1e-10))
    C1 = (0.01 * 1.0) ** 2
    C2 = (0.03 * 1.0) ** 2

    mu_x = torch.mean(dummy_data, dim=[2, 3])
    mu_y = torch.mean(gt_x, dim=[2, 3])

    sigma_x = torch.var(dummy_data, dim=[2, 3], unbiased=False)
    sigma_y = torch.var(gt_x, dim=[2, 3], unbiased=False)
    sigma_xy = torch.mean(dummy_data * gt_x, dim=[2, 3]) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    ssim = ssim_map.mean()
    print(mse, psnr, ssim)

    return dummy_data, rec_dummy_label


def idlg(model, gt_grads, dummy, rec_epochs=300, rec_lr=0.075, device="cpu"):
    """Improved Deep Leakage Gradients

    iDLG theoretically gives label prediction
    https://arxiv.org/pdf/2001.02610.pdf

    :param model: inferred model
    :param gt_grads: gradients of the ground truth data
    :param dummy: TorchDummy object
    :param rec_epochs: reconstruct epochs
    :param rec_lr: reconstruct learning rate
    :param device: cpu or cuda
    :return: dummy data, label prediction
    """
    # be careful about the influence on model.train() & model.eval()
    # especially dropout layer and bn layer are included in model

    dummy_data = dummy.generate_dummy_input(device)
    # extract ground-truth labels proposed by iDLG
    label_pred = torch.argmin(torch.sum(gt_grads[-2], dim=-1), dim=-1).detach().reshape((1,))

    optimizer = torch.optim.LBFGS([dummy_data], lr=rec_lr)
    criterion = nn.CrossEntropyLoss().to(device)

    pbar = tqdm(range(rec_epochs),
                total=rec_epochs,
                desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
    for _ in pbar:
        def closure():
            optimizer.zero_grad()
            model.zero_grad()

            dummy_pred = model(dummy_data)
            dummy_loss = criterion(dummy_pred, label_pred)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_diff = 0
            for dummy_g, origin_g in zip(dummy_dy_dx, gt_grads):
                grad_diff += ((dummy_g - origin_g) ** 2).sum()
            grad_diff.backward()

            return grad_diff

        loss = optimizer.step(closure)
        pbar.set_description("Loss {:.6}".format(loss))

    # save the dummy data
    dummy.append(dummy_data)
    # save the label prediction
    dummy.append_label(label_pred)


    return dummy_data, label_pred
