import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.PatchPooling import AveragePoolingWindow

class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # d_model:512,d_ff:2048
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x:(32,11,512),其余参数都是None
        # attention是AttentionLayer创建的对象
        # new_x:(32,11,512),作用完fullattention+MLP的输出
        # attn:None
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        # x:(32,11,512)
        x = x + self.dropout(new_x)
        # y:(32,11,512)
        y = x = self.norm1(x)
        # y:(32,11,512)->(32,512,11)
        # 卷积:(32,512,11)->(32,2048,11)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y:(32,2048,11)->(32,512,11)->(32,11,512)
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        # (32,11,512):相当于是输入x+注意力的输出+卷积层的输出
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, mode="None"):
        super(Encoder, self).__init__()
        # 是2个EncoderLayer构成的列表
        self.attn_layers = nn.ModuleList(attn_layers)
        # 默认是None
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        # 是torch.nn.LayerNorm(configs.d_model)
        self.norm = norm_layer
        self.mode = mode

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x:(32,11,512)
        attns = []
        # 不执行
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)


        else:
            window_list = [4,2,2]
            stride_list = [1,2,2]
            for index,attn_layer in enumerate(self.attn_layers):
                # 输入的x:(32,11,512)
                # attn_layer就是EncoderLayer创建的对象
                # 输出x:(32,11,512)
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                if "B" in self.mode:
                    patchpooling = AveragePoolingWindow(window=window_list[index],stride=stride_list[index])
                    x = patchpooling(x)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
