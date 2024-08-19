import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Dimfuse(nn.Module):

    def __init__(self, args):
        """
        D:原始时序变量个数
        """
        super(Dimfuse, self).__init__()
        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=args.enc_in, out_channels=args.out_channels,
                                                  kernel_size=args.kernel_size, stride=1, padding=0),
                                        nn.Conv1d(in_channels=args.out_channels, out_channels=args.enc_in,
                                                  kernel_size=args.kernel_size, stride=1, padding=0)
                                        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.pad = args.pad

    def forward(self, x):
        # 能不能修改成先残差再relu?
        res = x
        for conv in self.conv_list:
            x = F.pad(x, self.pad)
            x = conv(x)
            x = self.relu(x)
            x = self.dropout(x)
        x = x + res
        return x



if __name__ == "__main__":
    # 前向流程验证
    model1 = Dimfuse(kernel_size=16,pad=(0,15))
    input = torch.randn(32, 96, 7)  # (32,96,7)
    input = input.transpose(1, 2)  # (32,7,96)
    output = model1(input)  # (32,1,96)
    print(output.size())

