import torch
import BMMExt
import numpy as np

if __name__ == "__main__":
    a = torch.randn((4, 16, 3)).cuda()
    b = torch.randn((3, 3, 10)).cuda()
    c = torch.randn((3, 10)).cuda()
    s = torch.FloatTensor([16, 16, 32])
    res = torch.empty((4, 16, 10)).cuda()
    BMMExt.forward(a, b, s, c, res, 4, 16)
    
    bb = torch.empty((4, 3, 10))
    bb[0] = b[0]
    bb[1] = b[1]
    bb[2] = b[2]
    bb[3] = b[2]
    
    cc = torch.empty((4, 10))
    cc[0] = c[0]
    cc[1] = c[1]
    cc[2] = c[2]
    cc[3] = c[2]
    expected = torch.baddbmm(cc, a, bb)
    np.testing.assert_allclose(
                res.numpy().flatten(),
                expected.numpy().flatten(),
                rtol=1e-4
    )
