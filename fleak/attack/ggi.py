""" Inverting Gradients - How easy is it to break privacy in federated learning?

https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf
Caution: model.train() & model.eval() issue

 """

import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.autograd import Variable
import lpips

from .dlg import dummy_criterion
from ..model import MetaModel
from fleak.model.gan import CifarGenerator

def ggi(model, gt_grads, dummy, gt_x, rec_epochs=4000, rec_lr=0.1, tv=1e-6, device="cpu"):
    """Reconstruct an image from single gradients

    Similar to DLG based methods but adopting cosine similarity as the loss function
    Utilizing Adam other than LBFGS as the optimizer

    :param model: inferred model
    :param gt_grads: gradients of the ground-truth data
    :param dummy: TorchDummy object
    :param rec_epochs: number of reconstruct epochs
    :param rec_lr: reconstruct learning rate
    :param tv: hyperparameter for TV term
    :param device: cpu or cuda
    :return: dummy data & dummy label
    """
    model.eval()

    reconstructor = GradientReconstructor(model, dummy, rec_epochs, rec_lr, tv, device)
    # dummy_data, dummy_label = reconstructor.reconstruct(gt_grads)

    dummy_data, dummy_label = reconstructor.reconstruct_ggi(gt_grads)

    ##metrics

    #
    # img1 = copy.deepcopy(dummy_data)
    # img2 = copy.deepcopy(gt_x)
    # if img1.max() > 1.0 or img2.min() < -1.0:
    #     img1 = img1 / 255.0 * 2 - 1  # 从 [0, 2
    #     img2 = img2 / 255.0 * 2 - 1
    #
    # img1 = img1.to(device)
    # img2 = img2.to(device)
    #
    # loss_fn = lpips.LPIPS(net='alex')  # 'alex
    # loss_fn = loss_fn.to(device)
    #
    # lpips_distance = loss_fn(img1, img2)

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

    dummy.append(dummy_data)
    dummy.append_label(dummy_label)
    #
    # ##metrics
    # mse = (dummy_data.detach() - gt_x).pow(2).mean()


    return dummy_data, dummy_label


class GradientReconstructor:
    """Reconstruct an image from gradient after single step of gradient descent"""

    def __init__(self, model, dummy, epochs, lr, tv, device):
        """

        :param model: inferred model
        :param dummy: TorchDummy object
        :param epochs: reconstruct epochs
        :param lr: reconstruct learning rate
        :param tv: hyperparameter for TV term
        :param device: cpu or cuda
        """
        self.model = model
        self.dummy = dummy
        self.epochs = epochs
        self.lr = lr
        self.tv = tv
        self.device = device

        # if converting labels to integers
        self.convert_label = False

    def reconstruct(self, gt_grads):
        # generate dummy data with Gaussian distribution
        dummy_data = self.dummy.generate_dummy_input(self.device)

        # server has no access to inferred data labels
        if self.dummy.batch_size == 1:
            # label prediction by iDLG
            # dummy label is not updated
            dummy_label = torch.argmin(torch.sum(gt_grads[-2], dim=-1), dim=-1).detach().reshape((1,))
            optimizer = optim.Adam([dummy_data], lr=self.lr)
            criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.convert_label = True
            # DLG label recovery
            # dummy labels should be simultaneously updated
            dummy_label = self.dummy.generate_dummy_label(self.device)
            optimizer = optim.Adam([dummy_data, dummy_label], lr=self.lr)
            criterion = dummy_criterion

        # set learning rate decay
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[self.epochs // 2.667, self.epochs // 1.6, self.epochs // 1.142],
            gamma=0.1
        )  # 3/8 5/8 7/8

        pbar = tqdm(range(self.epochs),
                    total=self.epochs,
                    desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')
        for _ in pbar:
            closure = self._gradient_closure(optimizer, criterion, gt_grads, dummy_data, dummy_label)
            rec_loss = optimizer.step(closure)
            pbar.set_description("Rec. Loss {:.6}".format(rec_loss))
            scheduler.step()

            # small trick 2: project into image space
            with torch.no_grad():
                if self.dummy.normalize:
                    dummy_data.data = torch.max(
                        torch.min(dummy_data, (1 - self.dummy.t_dm) / self.dummy.t_ds),
                        -self.dummy.t_dm / self.dummy.t_ds)
                else:
                    dummy_data.data = torch.clamp(dummy_data, 0, 1)
        if self.convert_label:
            return dummy_data, torch.argmax(dummy_label, dim=-1)

        return dummy_data, dummy_label
    def reconstruct_ggi(self, gt_grads):
        # dummy_data_1 = self.dummy.generate_dummy_input(self.device)

        # generate dummy data with GAN
        generator = CifarGenerator().to(device=self.device)
        if self.dummy.batch_size == 1:
            path = r'D:/leakage-attack-in-federated-learning/models_parameter/GAN_cpa_1.pth'
            generator.load_state_dict(torch.load(path))
        elif self.dummy.batch_size == 64:
            path = r'D:/leakage-attack-in-federated-learning/models_parameter/GAN_cpa.pth'
            generator.load_state_dict(torch.load(path))
        elif self.dummy.batch_size == 30:
            path = r"D:/leakage-attack-in-federated-learning/models_parameter/GAN_cpa_50.pth"
            generator.load_state_dict(torch.load(path))
        elif self.dummy.batch_size == 50:
            path = r"D:/leakage-attack-in-federated-learning/models_parameter/GAN_cpa_50_real.pth"
            generator.load_state_dict(torch.load(path))

        z = Variable(torch.randn(self.dummy.batch_size, 100, 1, 1)).to(self.device)
        dummy_g = generator(z)
        dummy_data = dummy_g.detach()
        dummy_data.requires_grad = True

        # server has no access to inferred data labels
        if self.dummy.batch_size == 1:
            # label prediction by iDLG
            # dummy label is not updated
            dummy_label = torch.argmin(torch.sum(gt_grads[-2], dim=-1), dim=-1).detach().reshape((1,))
            optimizer = optim.Adam([dummy_data], lr=self.lr)
            criterion = nn.CrossEntropyLoss().to(self.device)
        else:
            self.convert_label = True
            # DLG label recovery
            # dummy labels should be simultaneously updated
            dummy_label = self.dummy.generate_dummy_label(self.device)
            optimizer = optim.Adam([dummy_data, dummy_label], lr=self.lr)
            criterion = dummy_criterion

        # # set learning rate decay
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[self.epochs // 2.667, self.epochs // 1.6, self.epochs // 1.142],
            gamma=0.1
        )  # 3/8 5/8 7/8


        pbar = tqdm(range(self.epochs),
                    total=self.epochs,
                    desc=f'{str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))}')


        for _ in pbar:
            closure = self._gradient_closure(optimizer, criterion, gt_grads, dummy_data, dummy_label)
            rec_loss = optimizer.step(closure)
            pbar.set_description("Rec. Loss {:.6}".format(rec_loss))
            scheduler.step()

            # small trick 2: project into image space
            with torch.no_grad():
                if self.dummy.normalize:
                    dummy_data.data = torch.max(
                        torch.min(dummy_data, (1 - self.dummy.t_dm) / self.dummy.t_ds),
                        -self.dummy.t_dm / self.dummy.t_ds)
                else:
                    dummy_data.data = torch.clamp(dummy_data, 0, 1)


        if self.convert_label:
            return dummy_data, torch.argmax(dummy_label, dim=-1)

        return dummy_data, dummy_label

    def _gradient_closure(self, optimizer, criterion, gt_grads, dummy_data, dummy_label):
        def closure():
            optimizer.zero_grad()
            self.model.zero_grad()

            dummy_pred = self.model(dummy_data)
            dummy_loss = criterion(dummy_pred, dummy_label)
            dummy_grads = torch.autograd.grad(dummy_loss, self.model.parameters(), create_graph=True)

            rec_loss = cosine_similarity_loss(dummy_grads, gt_grads)
            rec_loss += self.tv * total_variation(dummy_data)
            # rec_loss += wd_loss_grad_tuple(dummy_grads,gt_grads)
            rec_loss.backward()

            # small trick 1: convert the grad to 1 or -1
            dummy_data.grad.sign_()
            return rec_loss
        return closure


def cosine_similarity_loss(dummy_grads, grads):
    """ Compute cosine similarity value

    Compared to L2-norm loss, it can additionally capture the direction information

    :param dummy_grads: gradients of dummy data
    :param grads: gradients of the ground truth data
    :return: the loss value
    """
    # numerator
    nu = 0
    # denominator
    dn0 = 0
    dn1 = 0
    for dg, g in zip(dummy_grads, grads):
        # equivalent to the inner product of two vectors
        nu += (dg * g).sum()
        dn0 += dg.pow(2).sum()
        dn1 += g.pow(2).sum()
    loss = 1 - nu / dn0.sqrt() / dn1.sqrt()  # l2-norm
    return loss
def l2_norm(dummy_grads, grads):
    loss = 0
    for dg, g in zip(dummy_grads,grads):
        loss += ((dg - g) ** 2).sum()
    return loss


def total_variation(x):
    """ Total variation

    https://cbio.mines-paristech.fr/~jvert/svn/bibli/local/Rudin1992Nonlinear.pdf

     """
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def WD_Loss(fake_grad, gt_grad, num_bins=256):
    grad1_flat = fake_grad.view(-1)
    grad2_flat = gt_grad.view(-1)

    # 确定直方图的取值范围，根据两个梯度tensor的最小值和最大值
    min_val = min(grad1_flat.min(), grad2_flat.min()).item()
    max_val = max(grad1_flat.max(), grad2_flat.max()).item()

    # 计算直方图
    hist1 = torch.histc(grad1_flat, bins=num_bins, min=min_val, max=max_val)
    hist2 = torch.histc(grad2_flat, bins=num_bins, min=min_val, max=max_val)

    # 归一化直方图，使其和为1
    hist1 = hist1 / (hist1.sum() + 1e-9)
    hist2 = hist2 / (hist2.sum() + 1e-9)

    # 计算累积分布函数 (CDF)
    cdf1 = torch.cumsum(hist1, dim=0)
    cdf2 = torch.cumsum(hist2, dim=0)

    # 计算 bin 宽度
    bin_width = (max_val - min_val) / num_bins

    # 计算 WD 距离：累积直方图差值的 L1 距离乘以 bin 宽度
    wd = torch.sum(torch.abs(cdf1 - cdf2)) * bin_width
    return wd


def wd_loss_grad_tuple(fake_grads, gt_grads, num_bins=256, aggregate='mean'):
    """
    计算两个 tuple 中对应梯度 tensor 的 Wasserstein 距离（WD 距离），并对各个 WD 距离进行聚合。

    参数:
        tuple1 (tuple): 包含多个梯度 tensor 的 tuple。
        tuple2 (tuple): 包含多个梯度 tensor 的 tuple，要求与 tuple1 长度一致。
        num_bins (int): 用于构造直方图的分箱数量，默认值为256。
        aggregate (str): 聚合方式，'mean' 表示取平均，'sum' 表示求和。

    返回:
        torch.Tensor: 标量 tensor，表示两个 tuple 中所有对应梯度 tensor WD 距离的聚合值。
    """
    if len(fake_grads) != len(gt_grads):
        raise ValueError("输入的两个 tuple 长度必须一致")

    wd_list = []
    for g1, g2 in zip(fake_grads, gt_grads):
        wd_list.append(WD_Loss(g1, g2, num_bins))

    wd_tensor = torch.stack(wd_list)
    if aggregate == 'sum':
        return wd_tensor.sum()
    else:  # 默认采用 mean
        return wd_tensor.mean()

