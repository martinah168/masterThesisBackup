import torch


def psnr(img1, img2):
    """
    Args:
        img1: (n, c, h, w)
    """
    v_max = 1.
    # (n,)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    return 20 * torch.log10(v_max / torch.sqrt(mse))
