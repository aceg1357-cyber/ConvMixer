import torch
import torch.nn as nn
import torch.nn.init as init
class VariableFusionModule(nn.Module):
    def __init__(self):
        super(VariableFusionModule, self).__init__()
        self.conv_list = nn.ModuleList([nn.Conv2d(in_channels=7, out_channels=7, kernel_size=(1, 1)),
                                       ])
        # kaiming初始化
        for conv in self.conv_list:
            init.kaiming_uniform_(conv.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # 在变量维度上进行平均池化
        out = torch.mean(x, dim=1, keepdim=True)
        # 将输出维度调整为与输入相同的维度
        out = out.expand(-1, x.size(1), -1, -1)
        # 使用 1x1 卷积进行信息融合
        for conv in self.conv_list:
            out = conv(out)
            out = self.relu(out)
            out = self.dropout(out)
        return out

if __name__ == '__main__':
    input = torch.randn(32,7,8,11)
    model = VariableFusionModule()
    output = model(input)
    print(output.size())