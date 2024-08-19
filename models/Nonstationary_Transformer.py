import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.SelfAttention_Family import DSAttention, AttentionLayer
from layers.Embed import DataEmbedding
import torch.nn.functional as F


class Projector(nn.Module):
    '''
    MLP to learn the De-stationary factors
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    '''

    def __init__(self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3):
        """
        enc_in:7
        seq_len:96
        hidden_dims:[128,128]
        hidden_layers:2
        output_dim:1
        kernel_size:3
        """
        super(Projector, self).__init__()

        padding = 1 if torch.__version__ >= '1.5.0' else 2
        # 和TimesNet不同的是,TimesNet是将7转化为512,而这里是将96转化为1
        # (32,96,7)->(32,1,7)
        self.series_conv = nn.Conv1d(in_channels=seq_len, out_channels=1, kernel_size=kernel_size, padding=padding,
                                     padding_mode='circular', bias=False)
        # W(14,128)
        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        # [Linear(14,128),relu,Linear(128,128),relu,Linear(128,1)]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        """
        x:-------x_raw(32,96,7),表示未进行归一化的原始数据
        stats: --std_enc(32,1,7)
        """
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E # x:(32,96,7)->(32,1,7)
        x = torch.cat([x, stats], dim=1)  # B x 2 x E # x:(32,2,7)
        x = x.view(batch_size, -1)  # B x 2E # x:(32,14)
        y = self.backbone(x)  # B x O # y:(32，1)

        return y


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ucNDIDRNjjv
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                    output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                               configs.dropout)
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            DSAttention(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        AttentionLayer(
                            DSAttention(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        # tau_learner和delta_learner完全相同
        self.tau_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len, hidden_dims=configs.p_hidden_dims,
                                     hidden_layers=configs.p_hidden_layers, output_dim=1)
        self.delta_learner = Projector(enc_in=configs.enc_in, seq_len=configs.seq_len,
                                       hidden_dims=configs.p_hidden_dims, hidden_layers=configs.p_hidden_layers,
                                       output_dim=configs.seq_len)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        x_enc:(32,96,7)
        x_mark_enc:(32,96,4)
        x_dec:(32,144,7):0-48行是原始时序的48-96的数据,48-144为96个0
        x_mark_dec:(32,144,4):0-144行原始时序48-144行的周期特征
        """
        # detach()的作用是不计算梯度,但是生成的tensor是和原tensor共享参数空间的
        # clone的作用是创建副本,和原tensor不共享参数空间
        # 保证x_raw的变动不影响x_enc的变动,且x_raw不参与梯度计算
        x_raw = x_enc.clone().detach()
        # 标准化:Normalization
        # x_enc:(32,96,7)->(32,96,7)
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # B x S x E, B x 1 x E -> B x 1, positive scalar
        # 输入:x_raw(32,96,7),std_enc(32,1,7)
        # 输出:tau(32,1)
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        # 输入:x_raw(32,96,7),mean_enc(32,1,7)
        # 输出:delta(32，1)
        delta = self.delta_learner(x_raw, mean_enc)

        # 输入x_enc的倒数48行,即原始数据的48-96行,即(32,48,7)
        # 全零tensor:(32,96,7)
        # 合并得到x_dec_new(32,144,7)
        x_dec_new = torch.cat([x_enc[:, -self.label_len:, :], torch.zeros_like(x_dec[:, -self.pred_len:, :])],
                              dim=1).to(x_enc.device).clone()

        # 采用和TimesNet相同的DataEmbedding
        # enc_out = (32,96,512)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # 进行DSAttention
        # tau,delta:(32,1)
        # enc_out:(32,96,512)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # 构造decoder层的输入
        # x_dec_new:(32,144,7),x_mark_dec:(32,144,4)
        # 进行TimesNet的DataEmbedding
        # dec_out:(32,144,512)
        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)

        # 将上面的结果和encoder层的输出一起输入进decoder
        # 输入:dec_out:(32,144,512),enc_out:(32,96,512)
        # 输出:dec_out:(32,144,7),具体的操作先不看了
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta)

        # (32,144,7)*(32,1,7)+(32,1,7)
        # 输出:dec_out:(32,144,7)
        # 144可以理解为看到最后的48个数据,往后预测96个数据？
        # dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        mean_enc = mean_enc.unsqueeze(1).detach()
        x_enc = x_enc - mean_enc
        x_enc = x_enc.masked_fill(mask == 0, 0)
        std_enc = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        std_enc = std_enc.unsqueeze(1).detach()
        x_enc /= std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def anomaly_detection(self, x_enc):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        x_enc = x_enc / std_enc
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.projection(enc_out)
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        std_enc = torch.sqrt(
            torch.var(x_enc - mean_enc, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()  # B x 1 x E
        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc).exp()
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)
        # embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, L, D]
        return None
