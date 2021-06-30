import torch
import BMMExt
import numpy as np

if __name__ == "__main__":
    a = torch.ones((4, 16, 3)).cuda()
    b = torch.ones((3, 3, 10)).cuda()
    s = torch.FloatTensor([16, 16, 32])
    res = torch.zeros((4, 16, 10)).cuda()
    res = BMMExt.forward(a, b, s, res, 4, 16)
    
    bb = torch.empty((4, 3, 10)).cuda()
    bb[0] = b[0]
    bb[1] = b[1]
    bb[2] = b[2]
    bb[3] = b[2]
    
    expected = torch.bmm(a, bb)
    
    np.testing.assert_allclose(
                res.cpu().numpy().flatten(),
                expected.cpu().numpy().flatten(),
                rtol=1e-4
    )
