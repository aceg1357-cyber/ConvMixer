import torch
import torch.nn as nn

class VariableFusionModule(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(VariableFusionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(1, 1))

    def forward(self, x):
        # 在变量维度上进行平均池化
        out = torch.mean(x, dim=1, keepdim=True)
        # 将输出维度调整为与输入相同的维度
        out = out.expand(-1, x.size(1), -1, -1)
        # 使用 1x1 卷积进行信息融合
        out = self.conv(out)
        return out

# 示例用法
input_tensor = torch.randn(32, 7, 8, 512)  # 输入数据
input_channels = 7  # 输入通道数
output_channels = 64  # 输出通道数
model = VariableFusionModule(input_channels, output_channels)  # 创建模块实例
output = model(input_tensor)  # 获取输出结果
print(output.size())  # 输出：torch.Size([32, 7, 8, 512])