import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding
# from layers.ConvFuse import Dimfuse,TranposeDimfuse
from layers.abu_Convfuse import Dimfuse
from layers.TransformerBlock import TransformerBlock
from utils.tensor_transform import feature_seperator
from layers.ProductLayer import ProductLayer
from layers.FitBiasLayers import FitBiasLayers
from layers.ModernTCN import ModernTCN
from layers.TCN import TemporalConvNet
class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        n_vars:7
        nf:6144
        target_window:96
        head_dropout:0.1
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        """
        输入s:(32,7,512,12)
        """
        # x:(32,7,512,12)->(32,7,6144)
        x = self.flatten(x)
        # W(6144,96) x:(32,7,6144)->(32,7,96)
        x = self.linear(x)
        # (32,7,96)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.mode = configs.mode
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        padding = stride

        # patching and embedding
        # d_model:512
        # patch_len:16
        # stride:8
        # padding:8
        # dropout:0.1
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout,configs.mode)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            mode = self.mode
        )

        if "D" in self.mode:
            self.transformerblock = TransformerBlock(d_model=configs.d_model*configs.enc_in,nhead=8,dim_feedforward=4*configs.d_model*configs.enc_in,dropout=configs.dropout,n_layers=configs.e_layers)
        else:
            self.transformerblock = TransformerBlock(d_model=configs.d_model, nhead=8,
                                                     dim_feedforward=4 * configs.d_model,
                                                     dropout=configs.dropout, n_layers=configs.e_layers)
        # Prediction Head
        # head_nf:512*int((96-16)/8+2)=6144
        # self.head_nf = configs.d_model * \
        #                int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = 6144
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                self.head_nf * configs.enc_in, configs.num_class)

        self.dimfuse = Dimfuse(args=configs)
        if "A2" in self.mode:
            self.moderntcn = ModernTCN(configs.enc_in,configs.seq_len,configs.pred_len)
        if "A3" in self.mode:
            self.tcn = TemporalConvNet(configs.enc_in,[configs.enc_in])
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        只用到了e_enc:(32,96,7)
        """
        # Normalization from Non-stationary Transformer 归一化
        # x_enc:(32,96,7)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        # x_enc:(32,7,96)
        x_enc = x_enc.permute(0, 2, 1)

        # 是否添加模块A
        if "A" in self.mode:
            x_enc = self.dimfuse(x_enc)

        if "A2" in self.mode:
            x_enc = self.moderntcn(x_enc)

        if "A3" in self.mode:
            x_enc = self.tcn(x_enc)
        # u: [bs * nvars x patch_num x d_model]
        # enc_out:(224,12,512),注意这里的11表示长度96的序列被滑动窗口切分成了11个patch,而不是变量个数为7+4
        # 224可以理解为32个样本贡献7个特征,共224种"特征"
        # n_vars=7
        enc_out, n_vars = self.patch_embedding(x_enc)

        if "D" in self.mode:
            _,n,dim = enc_out.size()
            # (224,12,512)->(32,7,12,512)->(32,12,3584)
            enc_out = enc_out.view(int(_/7),7,n,dim).reshape(int(_/7),n,-1)

        # 仓库里的Encoder
        # z: [bs * nvars x patch_num x d_model]
        # 和itransformer一样,也是走FullAttention?这个地方真的和原论文是保持一致的吗？
        # enc_out:(224,11,512*7)
        enc_out, attns = self.encoder(enc_out)

        # 自己实现的TransformerEncoder
        # (32,12,3584)->(32,12,3584) or (224,12,512)->(224,12,512)
        # enc_out = self.transformerblock(enc_out)

        if "D" in self.mode:
            # (32,12,3584)->(32,7,12,512)
            # enc_out = enc_out.view(int(_/7),7,n,dim)
            enc_out = feature_seperator(enc_out,self.enc_in)

        else:
            # (224,12,512)->(32,7,12,512)
            enc_out = torch.reshape(
                enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # (32,7,12,512)->(32,7,512,12)
        # enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        # 输入:(32,7,512,12)
        # dec_out:(32,7,96)
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]

        # 转置卷积
        # if "A" in self.mode:
        #     dec_out = self.transposedimfuse(dec_out)
        # (32,7,96)->(32,96,7)
        dec_out = dec_out.permute(0, 2, 1)


        # De-Normalization from Non-stationary Transformer
        # 反归一化
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # 输出:(32,96,7)
        if "F" in self.mode:
            productlayer = ProductLayer()
            productlayer.to(self.device)
            # x_enc:(32,7,96)
            x_enc = x_enc.transpose(1,2) # (32,96,7)
            product_res = productlayer(x_enc) # (32,96,7)
            fitbiaslayers = FitBiasLayers(self.seq_len)
            fitbiaslayers.to(self.device)
            product_res = product_res.transpose(1,2) # (32,7,96)
            bias = fitbiaslayers(product_res) # (32,7,96)
            bias = bias.transpose(1,2)
            bias = bias * \
                      (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            bias = bias + \
                      (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + bias
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

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
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
