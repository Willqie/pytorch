import torch
import BMMExt
import numpy as np

if __name__ == "__main__":
    a = torch.rand((4, 128, 8)).cuda()
    b = torch.rand((3, 3, 8)).cuda()
    s = torch.FloatTensor([256, 128, 128])
    res = torch.zeros((4, 128, 3)).cuda()

    b_ = b.permute([0, 2, 1]).contiguous()
    res = BMMExt.op(a, b_, s, res, 4, 128)
    res = BMMExt.op(a, b_, s, res, 4, 128)
    
    
    print(b_.is_contiguous())
    bb = torch.empty((4, 8, 3)).cuda()
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
