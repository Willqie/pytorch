import torch
import BMMExt
import numpy as np

if __name__ == "__main__":
    a = torch.zeros((4, 128, 8)).cuda()
    b = torch.rand((3, 8, 3)).cuda()
    s = torch.FloatTensor([256, 128, 128])
    bias = torch.rand((3, 3)).cuda()
    bias_multiplier = torch.ones((128, )).cuda()
    res = torch.zeros((4, 128, 3)).cuda()

    b_ = b
    res = BMMExt.op(a, b_, s, res, bias, bias_multiplier, 4, 128)
    res = BMMExt.op(a, b_, s, res, bias, bias_multiplier, 4, 128)
    
    
    print(b_.is_contiguous())
    bb = torch.empty((4, 8, 3)).cuda()
    bb[0] = b_[0]
    bb[1] = b_[0]
    bb[2] = b_[1]
    bb[3] = b_[2]

    bias_ = torch.empty((4, 1, 3)).cuda()
    bias_[0] = bias[0]
    bias_[1] = bias[0]
    bias_[2] = bias[1]
    bias_[3] = bias[2]
    
    expected = torch.bmm(a, bb)
    expected += bias_
    # print(bias)
    
    np.testing.assert_allclose(
                res.cpu().numpy().flatten(),
                expected.cpu().numpy().flatten(),
                rtol=1e-4
    )
