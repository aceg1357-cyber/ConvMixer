import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from layers.OneConv import VariableFusionModule


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        # pe:(5000,512),且关闭梯度
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        # position:(5000,1)
        position = torch.arange(0, max_len).float().unsqueeze(1)\
        # div_term:(256,)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()
        # 赋值pe的奇数行和偶数行
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度
        # pe:(1,5000,512)
        pe = pe.unsqueeze(0)
        # register_buffer表示不参与参数更新的tensor能够被一起保存和加载
        self.register_buffer('pe', pe)

    def forward(self, x):
        # case1:TimesNet
        # x:(32,96,7)
        # return:(1,96,512),后续直接利用广播机制和(32,96,512)的x相加
        # case2:PatchTST
        # x:(224,11,16)
        # return:(1,11,512),后续直接利用广播机制和(224,11,512)的x相加
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    一维卷积构造TokenEmbedding(32,96,512)
    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # 将卷积层的参数用kaiming初始化,fan_in表示根据通道数进行缩放,卷积层的激活函数使用leaky-relu(光一个nn.Conv1d包含激活函数？)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        # x:batch_x:(32,96,7)
        # 卷积核:(in=7,out=512,size=3,stride=1,padding=1)
        # 一维卷积:卷积核的大小是全部特征的部分时刻,即512个(7,3)的卷积核
        # (32,96,7)->(32,7,96)->(32,512,96)->(32,96,512)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    config.embed_type!="timeF"的时候进行的编码
    """
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    config.embed_type="timeF"的时候进行的编码
    直接用线性层作用x_mark(32,96,4)得到(32,96,512)
    """
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        # 对应了df_stamp的构造逻辑,不同的freq对应的特征数为如下字典
        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        # (4,512)
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        # x:(32,96,4),w:(4,512)
        # output:(32,96,512)
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        # c_in---------config.enc_in:7
        # d_model------config.d_model:512
        # embed_type---config.embed:"TimeF"
        # freq---------config.freq:"h"
        # dropout------config.dropout:0.1
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # x--------batch_x:(32,96,7)
        # x_mark---batch_x_mark:(32,96,4)
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(
                x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x) # embedding层后加dropout


class DataEmbedding_inverted(nn.Module):
    """
    itransformer用到的倒置嵌入
    """
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        """
        c_in:96
        d_model:512
        """
        super(DataEmbedding_inverted, self).__init__()
        # w:(96,512)
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        """
        x:(32,96,7)
        x_mark:(32,96,4)
        """
        # x:(32,96,7)->(32,7,96)
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # cat的对象为(32,7,96)和(32,4,96),合并后为(32,11,96)
            # 作用linear(w:(96,512)),(32,11,96)->(32,11,512)
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x:(32,11,512)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout,mode):
        """
        PatchTST:
        d_model:512
        patch_len:16
        stride:8
        padding:8
        dropout:0.1
        Crossformer:
        d_model:512
        patch_len:12
        stride:12
        padding:0
        dropout:0
        """
        super(PatchEmbedding, self).__init__()
        # Patching
        self.mode = mode
        self.patch_len = patch_len # 16
        self.stride = stride # 8
        # 在右侧填充padding=8个长度,填充数值为每一行最后的元素值
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        # W:(16,512)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.oneconv = VariableFusionModule()

    def forward(self, x):
        # x:(32,7,96)
        # do patching
        # n_vars:7,表示变量数
        n_vars = x.shape[1]
        # (32,7,96)->(32,7,104) 在右侧填充padding=8个长度,填充数值为每一行最后的元素值
        x = self.padding_patch_layer(x)
        # unfold函数实现滑动窗口,在最后一个维度按照16的窗口大小8的步长能得到11个patch
        # (32,7,11,16)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        # 这个地方加入mode:C,即把(32,7,11,16)视为图像,进行1*1卷积融合通道信息
        if "C" in self.mode:
            x = self.oneconv(x)

        # (32*7,11,16)=(224,11,16)
        # 理解为一个batch中有224个特征,每个特征都对应了11个长度为16的patch？
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        # value_embedding(x):(224,11,16)->(224,11,512)
        # position_embedding(x):(224,11,16)->(1,11,512)
        # 注意,两者顺序不能换,前者先变到512维,才能和后者广播机制相加
        x = self.value_embedding(x) + self.position_embedding(x)
        # return:(224,11,512),n_vars=7
        return self.dropout(x), n_vars
