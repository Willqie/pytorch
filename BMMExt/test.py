import torch
import BMMExt

if __name__ == "__main__":
    a = torch.randn((4, 16, 3))
    b = torch.randn((3, 3, 10))
    c = torch.randn((3, 10))
    s = torch.FloatTensor([16, 16, 32])
    res = torch.empty((4, 16, 10))
    BMMExt.forward(a, b, s, c, res, 4, 16)
