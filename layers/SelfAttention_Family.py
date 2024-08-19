import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        mask_flag:False
        factor:1
        scale:None
        attention_dropout:0.1
        output_attention:False
        """
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries,keys,values:(32,96,8,64)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # tau:(32,1,1,1)
        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        # delta:(32,1,1,1)
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        # Q:(32,96,8,64)K(32,96,8,64)-->(32,8,96,96)
        # (32,8,96,96)*(32,1,1,1)+(32,1,1,1)-->(32,8,96,96)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        # 默认mask_flag=False
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 进行A和V的计算
        # A(32,8,96,96),V(32,96,8,64)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # 返回V(32,96,8,64)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        """
        这里的Full应该就是指不进行mask的attention
        mask_flag:False
        factor(attention factor):1
        scale:None
        attention_dropout:0.1
        output_attention:False
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        queries,keys,values:x:(32,11,8,64)
        attn_mask,tau,delta:None
        """
        # B:32,L:11,H:8,E:64
        B, L, H, E = queries.shape
        # S:11,D:64
        _, S, _, D = values.shape
        # self.scale是None,因此等于1/sqrt(64)=1/8,即transformer中的根号dk
        scale = self.scale or 1. / sqrt(E)
        # 计算注意力得分
        # 注意这里的注意力得分,是变量和变量之间的,而不是位置和位置之间的
        # (32,8,11,64)*(32,8,11,64)->(32,8,11,11)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # 默认不进行mask,也就是每一个点可以利用后面的时序信息
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        # A:(32,8,11,11)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # V:(32,8,11,11)*(32,11,8,64).transpose(1,2)->(32,8,11,64)->(32,11,8,64)
        V = torch.einsum("bhls,bshd->blhd", A, values)

        # 返回V:(32,11,8,64)
        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        """
        attention:FullAttention创建的对象
        d_model:config.d_model:512
        n_heads:config.n_heads:8
        """
        super(AttentionLayer, self).__init__()
        # 表示d_keys非空的时候,保留原有的值,否则被d_model//n_heads赋值
        # 由于d_keys是None,因此计算d_model//n_heads=512//8=64
        d_keys = d_keys or (d_model // n_heads)
        # 同理d_values=64
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        # Wq:(512,64*8)=(512,512)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # Wk:(512,64*8)=(512,512)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        # Wv:(512,64*8)=(512,512)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        # W:(512,512)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        下面的注释都是基于PatchTST的
        queries,keys,values:x:(32,11,512)
        attn_mask,tau,delta:None

        前半部分是处理多头的变换逻辑
        后半部分的inner_attention是不同模型对应不同的Attention
        比如FullAttention,DSAttention
        """
        # B:32,L:11,S:11,H:8
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # queries:(32,11,512)->(32,11,512)->(32,11,8,64)
        queries = self.query_projection(queries).view(B, L, H, -1)
        # keys:(32,11,512)->(32,11,512)->(32,11,8,64)
        keys = self.key_projection(keys).view(B, S, H, -1)
        # values:(32,11,512)->(32,11,512)->(32,11,8,64)
        values = self.value_projection(values).view(B, S, H, -1)

        # 输入queries,keys,values均为(32,11,8,64)
        # 进行fullattention计算得到:(32,11,8,64)
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        # (32,11,8,64)->(32,11,512)
        out = out.view(B, L, -1)
        # 再作用一个MLP
        # (32,11,512)->(32,11,512),attn:None
        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        """
        seg_num:8
        factor:1
        d_model:512
        n_heads:8
        """
        super(TwoStageAttentionLayer, self).__init__()
        # d_ff:512
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        # 可学习参数:(8,1,512)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        """
        x:(32,7,8,512)
        """
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        # time_in:(32,7,8,512)->(224,8,512)
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        # time_enc:(224,8,512)->(224,8,512)
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        # Dropout+Add+LN+MLP+Dropout+Add+LN
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        # 输入:dim_in:(224,8,512)
        # 输出:dim_send:(256,7,512)
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        # self.router:可学习参数:(8,1,512)
        # batch_router:(256,1,512)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        # 输入:batch_router:(256,1,512) dim_send:(256,7,512),分别表示Q,K,V,Q和KV不同源
        # 输出dim_buffer:(256,1,512)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        # 输入:dim_send:(256,7,512),dim_buffer:(256,1,512) ,分别表示Q,K,V,Q和KV不同源
        # 输出:dim_receive:(256,7,512)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)

        # 目前的2个输出:
        # dim_send:(256,7,512),dim_buffer:(256,7,512)
        # 进行Dropout+LN+Add
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        # dim_enc:(256,7,512)->final_out:(32,7,8,512)
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
