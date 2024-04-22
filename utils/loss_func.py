import torch
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from itertools import combinations
import pdb


class NLLSurvLoss(nn.Module):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    alpha: float
        TODO: document
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    """

    def __init__(self, alpha=0.0, eps=1e-7, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.reduction = reduction

    def __call__(self, h, y, t, c):
        """
        Parameters
        ----------
        h: (n_batches, n_classes)
            The neural network output discrete survival predictions such that hazards = sigmoid(h).
        y_c: (n_batches, 2) or (n_batches, 3)
            The true time bin label (first column) and censorship indicator (second column).
        """

        return nll_loss(h=h, y=y.unsqueeze(dim=1), c=c.unsqueeze(dim=1),
                        alpha=self.alpha, eps=self.eps,
                        reduction=self.reduction)


# TODO: document better and clean up
def nll_loss(h, y, c, alpha=0.0, eps=1e-7, reduction='sum'):
    """
    The negative log-likelihood loss function for the discrete time to event model (Zadeh and Schmid, 2020).
    Code borrowed from https://github.com/mahmoodlab/Patch-GCN/blob/master/utils/utils.py
    Parameters
    ----------
    h: (n_batches, n_classes)
        The neural network output discrete survival predictions such that hazards = sigmoid(h).
    y: (n_batches, 1)
        The true time bin index label.
    c: (n_batches, 1)
        The censoring status indicator.
    alpha: float
        The weight on uncensored loss 
    eps: float
        Numerical constant; lower bound to avoid taking logs of tiny numbers.
    reduction: str
        Do we sum or average the loss function over the batches. Must be one of ['mean', 'sum']
    References
    ----------
    Zadeh, S.G. and Schmid, M., 2020. Bias in cross-entropy-based training of deep survival networks. IEEE transactions on pattern analysis and machine intelligence.
    """
    y = y.type(torch.int64)
    c = c.type(torch.int64)

    hazards = torch.sigmoid(h)

    S = torch.cumprod(1 - hazards, dim=1)

    S_padded = torch.cat([torch.ones_like(c), S], 1)

    # TODO: document/better naming
    s_prev = torch.gather(S_padded, dim=1, index=y).clamp(min=eps)
    h_this = torch.gather(hazards, dim=1, index=y).clamp(min=eps)
    s_this = torch.gather(S_padded, dim=1, index=y + 1).clamp(min=eps)

    uncensored_loss = -(1 - c) * (torch.log(s_prev) + torch.log(h_this))
    censored_loss = - c * torch.log(s_this)

    neg_l = censored_loss + uncensored_loss
    if alpha is not None:
        loss = (1 - alpha) * neg_l + alpha * uncensored_loss
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("Bad input for reduction: {}".format(reduction))

    return loss


class UniLoss(nn.Module):
    def __init__(self, reduction='mean', temperature=0.08):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def __call__(self, modalA, modalB, logitsA, logitsB, label):

        modalA = torch.mean(modalA, dim=1)
        modalB = torch.mean(modalB, dim=1)
        modalA = modalA / modalA.norm(dim=-1, keepdim=True)
        modalB = modalB / modalB.norm(dim=-1, keepdim=True)
        Y_hatA = torch.topk(logitsA, 1, dim=1)[1]
        Y_hatB = torch.topk(logitsB, 1, dim=1)[1]

        return uniconloss(modalA=modalA, modalB=modalB, Y_hatA=Y_hatA, Y_hatB=Y_hatB,
                          label=label.unsqueeze(1), temperature=self.temperature, reduction=self.reduction)


def uniconloss(modalA, modalB, Y_hatA, Y_hatB, label, temperature=0.07, reduction='sum'):
    modalA_ = modalA.detach()
    modalB_ = modalB.detach()

    A_Bool = Y_hatA.eq(label)
    A_Bool_ = ~ A_Bool
    B_Bool = Y_hatB.eq(label)
    B_Bool_ = ~B_Bool

    A_B_Bool = torch.gt(A_Bool | B_Bool, 0)
    A_B_Bool_ = torch.gt(A_Bool & B_Bool, 0)

    A_ = A_Bool_ | A_B_Bool_
    B_ = B_Bool_ | A_B_Bool_

    if True not in A_B_Bool:
        A_B_Bool = ~A_B_Bool
        A_ = ~A_
        B_ = ~B_
    mask = A_B_Bool.float()

    modalA_list = [modalA[i].clone() for i in range(modalA.shape[0])]
    for i in range(modalA.shape[0]):
        if not A_[i]:
            modalA_list[i] = modalA_[i].clone()
    modalA_list = torch.stack(modalA_list)

    modalB_list = [modalB[i].clone() for i in range(modalB.shape[0])]
    for i in range(modalB.shape[0]):
        if not B_[i]:
            modalB_list[i] = modalB_[i].clone()
    modalB_list = torch.stack(modalB_list)

    logits = torch.div(torch.matmul(modalA_list, modalB_list.T), temperature)
    #logits = torch.div(torch.matmul(modalA_list, modalB_list.permute(0, 2, 1)), temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)

    exp_logits = torch.exp(logits - logits_max.detach())[0]
    #masked_exp_logits = mask.unsqueeze(2) * exp_logits

    #mean_log_pos = -torch.log(masked_exp_logits.sum() / exp_logits.sum(dim=1).sum(dim=1) / mask.sum())
    mean_log_pos = - torch.log(((mask * exp_logits).sum() / exp_logits.sum()) / mask.sum())  # + 1e-6


    return mean_log_pos

def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0
    l1_reg += l1_reg_all(model.cell_networks)
    l1_reg += l1_reg_all(model.attention_network)

    return l1_reg