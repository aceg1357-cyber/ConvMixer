import torch
import torch.nn as nn
class PPNet(nn.Module):
    def __init__(self,seq_len):
        super().__init__()
        self.linear_1 = nn.Linear(seq_len,seq_len)
        self.linear_2 = nn.Linear(seq_len, seq_len)
        self.linear_3 = nn.Linear(seq_len, seq_len)
        self.linear_4 = nn.Linear(seq_len, seq_len)

    def forward(self,input1,input2):
        # 假设原始的tensor是x，shape为(96,7)
        x = torch.randn(96, 7)

        # 为了使每个列向量都与其它7个列向量计算哈达玛积，需要对x进行扩展，使其shape变为(96,7,7)
        x_expanded = x.unsqueeze(-1).expand(-1, -1, x.size(-1))

        # 由于两个矩阵执行哈达玛积运算需要它们具有相同的shape，因此对扩展后的x矩阵进行转置
        x_transposed = x_expanded.transpose(1, 2)

        # 计算哈达玛积
        result = x_expanded * x_transposed

        # 将结果汇总到原始的shape（96,7）
        result = result.sum(dim=1)

        print(result.shape)  # 输出为：torch.Size([96, 7])
        return output



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    input = torch.randn(32,7,96)
    model = PPNet(seq_len=96)
    output = model(input,input)
    print(output.size())
    # 假设你的模型名为 model
    model_parameters = count_parameters(model)
    print(f'该模型的参数量为: {model_parameters}')