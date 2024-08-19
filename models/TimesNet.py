import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    输入:x(32,192,512)
    返回:
    period:(1,5)[192//index1,192//index2,...,192//index5]
    abs(xf).mean(-1)[:, top_list]:(32,5),表示每个样本的最大的五个分量位置的特征均值
    """
    # [B, T, C]
    # (32,192,512)->(32,97,512)[97=192/2+1]
    # 对于每个特征,将长度为192的时序信号转化为97个频域上的信号
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes(振幅)
    # (32,97,512)->(97,512)->(97)
    # 取模表示计算信号的振幅
    # frequency_list表示97个实数值,并且将第一个设置为0
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    # 找到最大的5个分量的位置索引
    _, top_list = torch.topk(frequency_list, k)
    # top_list转numpy
    top_list = top_list.detach().cpu().numpy()
    # 192//[index1,index2,index3,index4,index5]等价于[192//index1,...,192//index5]
    period = x.shape[1] // top_list
    # (32,97,512)->(32,97)->(32,5)
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len    # 96
        self.pred_len = configs.pred_len  # 96
        self.k = configs.top_k            # 5
        # parameter-efficient design
        # (32,)
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        """
        输入x:(32,192,512)
        """
        # (32,192,512)
        B, T, N = x.size()
        # period_list:(1,5):例如[32,3,64,2,2](192//index得到)
        # period_weight:(32,5)
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            # period表示周期？
            period = period_list[i]
            # 假设period=33
            if (self.seq_len + self.pred_len) % period != 0:
                # length = 198
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                # padding:(32,6,512)
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                # out:(32,198,512)
                out = torch.cat([x, padding], dim=1)
            else:
                # length=192
                length = (self.seq_len + self.pred_len)
                out = x
            # case1:out(32,192,512)
            # 假设period=32,(32,192,512)->(32,6,32,512)->(32,512,6,32)
            # 表示每个特征下都有一个6*32的矩阵,表示将192的序列根据周期32拆分为了6段,即1D转2D的过程
            # case2:out(32,198,512)
            # (32,198,512)->(32,6,33,512)->(32,512,6,33)
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            # (32,512,6,32)->(32,2048,6,32)->(32,512,6,32)
            out = self.conv(out)
            # (32,512,6,32)->(32,6,32,512)->(32,192,512)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # 这一步和直接append(out)有什么区别？
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        # 5*(32,192,512)->(32,192,512,5)即每个特征下每个序列的5个最显著频域信号表征
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        # (32,5)->(32,5)
        period_weight = F.softmax(period_weight, dim=1)
        # (32,5)->(32,1,5)->(32,1,1,5)->(32,192,512,5)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        # (32,192,512,5)->(32,192,512)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        # res(32,192,512)
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # (96,192)的线性层
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            # (512*96,)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        # x_enc:(32,96,7),x_mark_enc:(32,96,4)
        # means:(32,1,7),去均值
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        # stdev:(32,1,7),去标准差
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5) # 无偏方差+微小量+开根号得到标准差
        # x_enc:(32,96,7)
        x_enc /= stdev

        # embedding:对特征维度进行融合
        # enc_out:(32,96,512)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]

        # 对时间维度进行融合
        # w:(96,192)
        # enc_out:(32,96,512)->(32,512,96)->(32,512,192)->(32,192,512)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension

        # TimesNet
        for i in range(self.layer): # 默认2层
            # 作用TimesBlock后进行LayerNorm
            # (32,192,512)->(32,192,512)[LN和TimesBlock都不改变size]
            # enc_out:(32,192,512)
            enc_out = self.layer_norm(self.model[i](enc_out))

        # project back
        # w:(512,7) dec_out:(32,192,7)
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        # (32,1,7)->(32,7)->(32,1,7)->(32,192,7)
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        # (32,1,7)->(32,7)->(32,1,7)->(32,192,7)
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x_enc------------batch_x:(32,96,7)
        # x_mark_enc-------batch_x_mark:(32,96,4)
        # x_dec------------dec_inp:(32,144,7)
        # x_mark_dec-------batch_y_mark:(32,144,4)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # dec_out:(32,192,7)
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # 返回:(32,96,7)(即取出了dec_out的倒数96行)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
