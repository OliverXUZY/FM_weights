import torch
import random
import numpy as np
import torch.nn.functional as F


def set_random_seed(seed_value=42):
    """
    Set random seed for reproducibility.
    
    Parameters:
    - seed_value (int): Value of the seed.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def intermdiates_to_pos_ctx(h_total, normalize = True):
    assert h_total.dim() == 4   #  [Layer, Batch*sets, T(Seq), D][13, 64, 197, 192]
    L, B, T, D = h_total.shape  # [13, 64, 197, 192]

    mu = h_total.mean(dim=(-1, -2, -3))

    expanded_mu_pos = mu.unsqueeze(1).unsqueeze(2).expand(-1, T, D)  # [Layer, T(Seq), D] [13,97,192]
    pos = h_total.mean(1) - expanded_mu_pos  # [Layer, T(Seq), D] [13,97,192]
    if normalize:
        pos = F.normalize(pos,dim = -1)

    expanded_mu_ctx = mu.unsqueeze(1).unsqueeze(2).expand(-1, B, D) # [Layer, Batch*sets, D] [13, 64 ,192]
    ctx = h_total.mean(2) - expanded_mu_ctx # [Layer, Batch*sets, D] [13, 64 ,192]
    if normalize:
        ctx = F.normalize(ctx,dim = -1)

    expanded_pos = pos.unsqueeze(1).expand(-1, B, -1, -1)   #  [Layer, Batch*sets, T(Seq), D][13, 64, 197, 192]
    expanded_ctx = ctx.unsqueeze(2).expand(-1, -1, T, -1)   #  [Layer, Batch*sets, T(Seq), D][13, 64, 197, 192]
    expanded_mu_whole = mu.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(-1, B, T, D) 

    resid = h_total - expanded_mu_whole - expanded_pos - expanded_ctx      #  [Layer, Batch*sets, T(Seq), D][13, 64, 197, 192]


    cvec = expanded_ctx + resid  #  [Layer, Batch*sets, T(Seq), D][13, 64, 197, 192]

    retval = {
        "mu": mu,
        "pos": pos,
        "ctx": ctx,
        "resid": resid,
        "cvec": cvec
    }
    return retval







