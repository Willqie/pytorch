import torch
import BMMExt
import numpy as np

if __name__ == "__main__":
    a = torch.rand((4, 128, 8)).cuda()
    b = torch.rand((3, 3, 8)).cuda()
    s = torch.FloatTensor([256, 128, 128])
    res = torch.zeros((4, 128, 3)).cuda()
    res = BMMExt.op(a, b.permute([0, 2, 1]), s, res, 4, 128)
    res = BMMExt.op(a, b.permute([0, 2, 1]), s, res, 4, 128)
    
    b_ = b.permute([0,2,1])
    bb = torch.empty((4, 3, 8)).cuda()
    bb[0] = b_[0]
    bb[1] = b_[0]
    bb[2] = b_[1]
    bb[3] = b_[2]
    
    expected = torch.bmm(a, bb)
    
    np.testing.assert_allclose(
                res.cpu().numpy().flatten(),
                expected.cpu().numpy().flatten(),
                rtol=1e-4
    )
