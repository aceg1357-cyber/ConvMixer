import torch
import torch.nn as nn
from einops import rearrange, repeat
from layers.SelfAttention_Family import TwoStageAttentionLayer


class SegMerging(nn.Module):
    """
    将(batch,dim,patch_num,length)进行重组
    例如(32,7,8,512)->2个(32,7,4,512)->(32,7,4,1024)
    并且最后两个维度构成的矩阵中,第一行表示第1和第4个patch的合并,最后一行是第5个第8个patch的合并
    """
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        """
        d_model:512
        win_size:2
        """
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        """
        x:(32,7,8,12)
        """
        # 32,7,8,12
        batch_size, ts_d, seg_num, d_model = x.shape
        # pad_num = 0
        pad_num = seg_num % self.win_size
        # 默认不需要pad
        if pad_num != 0:
            # 如果需要pad,举例:seg_num=7,win_size=2,则pad_num=1,就会把最后一个patch放到末尾填充,凑够8个patch
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, :, -pad_num:, :]), dim=-2)

        # 得到2个(32,7,4,512)
        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, :, i::self.win_size, :])
        # 2个(32,7,4,512)构成(32,7,4,1024)
        x = torch.cat(seg_to_merge, -1)
        # LN:(32,7,4,1024)
        x = self.norm(x)
        # x:(32,7,4,1024)->(32,7,4,512)
        x = self.linear_trans(x)

        return x


class scale_block(nn.Module):
    def __init__(self, configs, win_size, d_model, n_heads, d_ff, depth, dropout, \
                 seg_num=10, factor=10):
        """
        configs:configs
        win_size:如果是第一个scale_block就是1,否则就是win_size=2
        d_model:512
        h_heads:8
        d_ff:2048
        depth:1,表示TwoStageAttentionLayer的层数
        dropout:0.1
        seg_num:如果是第一个scale_block就是self.seg_num=8,否则就是ceil(in_seg_num / win_size ** l)=8(l最多取到1)
        --------l=0: seg_num=8
        --------l=1: seg_num=4
        factor:1
        """
        super(scale_block, self).__init__()

        # 除了第一个scale_block以外,都需要进行SegMerging
        if win_size > 1:
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        # 第一个scale_block就是None
        else:
            self.merge_layer = None

        self.encode_layers = nn.ModuleList()

        for i in range(depth):
            self.encode_layers.append(TwoStageAttentionLayer(configs, seg_num, factor, d_model, n_heads, \
                                                             d_ff, dropout))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x:(32,7,8,512)
        """
        _, ts_dim, _, _ = x.shape

        if self.merge_layer is not None:
            x = self.merge_layer(x)

        for layer in self.encode_layers:
            x = layer(x)

        return x, None


class Encoder(nn.Module):
    def __init__(self, attn_layers):
        super(Encoder, self).__init__()
        self.encode_blocks = nn.ModuleList(attn_layers)

    def forward(self, x):
        """
        x_enc:(32,7,8,512)
        """
        encode_x = []
        encode_x.append(x)

        for block in self.encode_blocks:
            x, attns = block(x)
            encode_x.append(x)

        return encode_x, None


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, seg_len, d_model, d_ff=None, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_model),
                                  nn.GELU(),
                                  nn.Linear(d_model, d_model))
        self.linear_pred = nn.Linear(d_model, seg_len)

    def forward(self, x, cross):
        batch = x.shape[0]
        x = self.self_attention(x)
        x = rearrange(x, 'b ts_d out_seg_num d_model -> (b ts_d) out_seg_num d_model')

        cross = rearrange(cross, 'b ts_d in_seg_num d_model -> (b ts_d) in_seg_num d_model')
        tmp, attn = self.cross_attention(x, cross, cross, None, None, None,)
        x = x + self.dropout(tmp)
        y = x = self.norm1(x)
        y = self.MLP1(y)
        dec_output = self.norm2(x + y)

        dec_output = rearrange(dec_output, '(b ts_d) seg_dec_num d_model -> b ts_d seg_dec_num d_model', b=batch)
        layer_predict = self.linear_pred(dec_output)
        layer_predict = rearrange(layer_predict, 'b out_d seg_num seg_len -> b (out_d seg_num) seg_len')

        return dec_output, layer_predict


class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.decode_layers = nn.ModuleList(layers)


    def forward(self, x, cross):
        final_predict = None
        i = 0

        ts_d = x.shape[1]
        for layer in self.decode_layers:
            cross_enc = cross[i]
            x, layer_predict = layer(x, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1

        final_predict = rearrange(final_predict, 'b (out_d seg_num) seg_len -> b (seg_num seg_len) out_d', out_d=ts_d)

        return final_predict
