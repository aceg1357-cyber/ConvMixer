import torch
import torch.nn as nn

class AveragePoolingWindow(nn.Module):
    def __init__(self, window, stride):
        super(AveragePoolingWindow, self).__init__()
        self.window = window
        self.stride = stride

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        output_seq_len = (seq_len - self.window) // self.stride + 1
        pooled_output = []
        for i in range(0, seq_len - self.window + 1, self.stride):
            window_data = x[:, i:i + self.window, :]
            window_mean = torch.mean(window_data, dim=1)
            pooled_output.append(window_mean)

        return torch.stack(pooled_output, dim=1)

if __name__ == "__main__":
    # 使用示例
    input_data = torch.randn(32,11,512)# 输入数据
    print(input_data.size())
    window_size = 4
    stride = 1
    model = AveragePoolingWindow(window_size, stride)  # 创建模型实例
    output = model(input_data)  # 获取输出结果
    print(output.size())  # 输出：torch.Size([32, 8, 512])

