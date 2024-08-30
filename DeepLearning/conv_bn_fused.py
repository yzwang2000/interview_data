from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn

def main():
    torch.random.manual_seed(0)

    f1 = torch.randn(1, 2, 5, 5)
    module = nn.Sequential(OrderedDict(
        conv=nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False),
        bn=nn.BatchNorm2d(num_features=4)
    ))

    module.eval()
    with torch.no_grad():
        output1 = module(f1)
        print(output1)

    # Conv 参数
    kernel = module.conv.weight             # [4, 2, 3, 3]

    # bn 参数
    running_mean = module.bn.running_mean    # [4] 
    running_var = module.bn.running_var      # [4]
    gamma = module.bn.weight                 # [4]
    beta = module.bn.bias                    # [4]
    eps = module.bn.eps                      # 1e-5
    
    # 融合后的参数
    std = (running_var + eps).sqrt()         # [4]
    t = (gamma / std).reshape(-1, 1, 1, 1)   # [ch] -> [ch, 1, 1, 1]
    kernel = kernel * t                      # [ch] -> [ch, 1, 1, 1]
    bias = beta - running_mean * gamma / std # [ch]

    # 开始融合
    fused_conv = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1, bias=True)
    fused_conv.load_state_dict(OrderedDict(weight=kernel, bias=bias))

    with torch.no_grad():
        output2 = fused_conv(f1)
        print(output2)

    np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
    print("convert module has been tested, and the result looks good!")

if __name__ == '__main__':
    main()
