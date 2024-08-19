import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self,d_model, nhead, dim_feedforward, dropout,n_layers):
        super(TransformerBlock,self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, n_layers)

    def forward(self,x):
        x = self.transformer_encoder(x)
        return x

if __name__ == '__main__':
    input = torch.rand(32,11,512)
    model = TransformerBlock(512,8,2048,0.1,3)
    output = model(input)
    print(output.size())