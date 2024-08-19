import torch
import torch.nn as nn
class FitBiasLayers(nn.Module):
    def __init__(self,seq_len):
        super(FitBiasLayers, self).__init__()
        self.linear = nn.Linear(seq_len,seq_len)


    def forward(self, x):
        # 输入x:(32,96,7)是经过Product得到的特征交叉tensor
        # 希望用这个tensor去拟合单通道结果与真实结果的误差
        output = self.linear(x)
        return output

if __name__ == '__main__':
    model = FitBiasLayers(seq_len=96)
    x = torch.randn(32,7,96)
    output = model(x)
    print(output.size())
