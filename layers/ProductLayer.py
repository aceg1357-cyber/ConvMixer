import torch
import torch.nn as nn


class ProductLayer(nn.Module):
    def __init__(self):
        super(ProductLayer, self).__init__()

    def forward(self, x):
        x_expanded = x.unsqueeze(2).expand(-1, -1, x.size(-1), -1)
        x_transposed = x.unsqueeze(2).expand(-1, -1, x.size(-1), -1).transpose(2, 3)
        result = x_expanded * x_transposed
        result = result.sum(dim=2)
        return result


if __name__ == '__main__':
    input = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.float32)
    model = ProductLayer()
    output = model(input)
    print(output.size(),output)
