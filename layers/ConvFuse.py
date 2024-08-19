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
        # self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=i, out_channels=i - 6,
        #                                           kernel_size=3, stride=1, padding=1) for i in
        #                                 range(D, 1, -6)])
        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=args.in_channels, out_channels=args.out_channels,
                                                  kernel_size=args.kernel_size, stride=1, padding=0),
                                        nn.Conv1d(in_channels=args.out_channels, out_channels=args.in_channels,
                                                  kernel_size=args.kernel_size, stride=1, padding=0)
                                        ])
        # self.bn_list = nn.ModuleList([nn.BatchNorm1d(num_features=16),
        #                               nn.BatchNorm1d(num_features=7)])
        # kaiming初始化
        # for conv in self.conv_list:
        #     init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=args.dropout_rate)
        self.args = args

    def forward(self, x):
        # 能不能修改成先残差再relu?
        res = x
        for conv in self.conv_list:
            x = F.pad(x, self.args.pad)
            x = conv(x)
            if self.args.af == "relu":
                x = self.relu(x)
            else:
                x = self.gelu(x)
            x = self.dropout(x)
        x = x + res
        return x



class TranposeDimfuse(nn.Module):
    def __init__(self, D):
        """
        D:原始时序变量个数
        """
        super(TranposeDimfuse, self).__init__()
        # self.conv_list = nn.ModuleList([nn.ConvTranspose1d(in_channels=i, out_channels=i + 6,
        #                                                    kernel_size=3, stride=1, padding=1) for i in
        #                                 range(1, D, 6)])
        self.conv_list = nn.ModuleList([nn.Conv1d(in_channels=7, out_channels=16,
                                                           kernel_size=7, stride=1, padding=3),
                                        nn.Conv1d(in_channels=22, out_channels=21,
                                                           kernel_size=7, stride=1, padding=3)
                                        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
            # x = self.relu(x)
            # x = self.dropout(x)
        return x


if __name__ == "__main__":
    # 前向流程验证
    model1 = Dimfuse()
    input = torch.randn(32, 96, 7)  # (32,96,7)
    input = input.transpose(1, 2)  # (32,7,96)
    output = model1(input)  # (32,1,96)
    print(output.size())

