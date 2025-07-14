# RoFormer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. 旋转位置编码 (Rotary Positional Embeddings, RoPE) ---
# 参考实现：https://github.com/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
# 稍作修改以适应 RoFormer 的原始论文思路


def rotate_half(x):
    """
    将输入张量x的后半部分旋转到前半部分，并取负。
    例如：[x0, x1, x2, x3] -> [-x2, -x3, x0, x1]
    用于RoPE中的复数乘法模拟。
    """
    x = x.reshape(x.shape[:-1] + (2, x.shape[-1] // 2))
    x0, x1 = x.unbind(dim=-2)
    return torch.cat((-x1, x0), dim=-1)


def apply_rotary_pos_emb(q, k, cos_cached, sin_cached):
    """
    对Query (q) 和 Key (k) 应用旋转位置编码。
    Args:
        q (Tensor): Query张量 (batch_size, num_heads, seq_len, head_dim)
        k (Tensor): Key张量 (batch_size, num_heads, seq_len, head_dim)
        cos_cached (Tensor): 预计算的cos值 (1, 1, seq_len, head_dim)
        sin_cached (Tensor): 预计算的sin值 (1, 1, seq_len, head_dim)
    Returns:
        tuple: 应用RoPE后的 (q_rotated, k_rotated)
    """
    # 确保cos_cached和sin_cached的维度与q/k匹配，以便广播
    # 通常cos_cached和sin_cached是 (max_seq_len, head_dim) 或 (seq_len, head_dim)
    # 我们需要将其扩展到 (1, 1, seq_len, head_dim)

    # 假设cos_cached和sin_cached的形状是 (seq_len, head_dim)
    # 扩展维度以匹配 (batch_size, num_heads, seq_len, head_dim)
    cos_cached = cos_cached.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin_cached = sin_cached.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)

    q_rotated = (q * cos_cached) + (rotate_half(q) * sin_cached)
    k_rotated = (k * cos_cached) + (rotate_half(k) * sin_cached)
    return q_rotated, k_rotated


class RotaryPositionalEmbedding(nn.Module):
    """
    预计算和缓存旋转位置编码的cos和sin值。
    """

    def __init__(self, head_dim, max_seq_len=512, base=10000):
        super().__init__()
        # 计算旋转频率
        # inv_freq = 1 / (base^(i / d)) for i in [0, d/2-1]
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)  # 注册为buffer，不作为模型参数保存

        self.max_seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cos_sin_tables(self, x, seq_len):
        """
        根据当前序列长度更新或生成cos/sin表。
        """
        # x: (batch_size, num_heads, seq_len, head_dim)
        # 生成位置索引 [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)

        # freqs: (seq_len, head_dim / 2) * (head_dim / 2) -> (seq_len, head_dim / 2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # complex_freqs: (seq_len, head_dim / 2) -> (seq_len, head_dim)
        # [cos(f0), sin(f0), cos(f1), sin(f1), ...]
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)

        self.cos_cached = emb.cos().to(x.dtype)
        self.sin_cached = emb.sin().to(x.dtype)
        self.max_seq_len_cached = seq_len

    def forward(self, q, k):
        """
        获取适用于当前q, k的cos和sin值。
        """
        seq_len = q.shape[-2]  # q的倒数第二个维度是序列长度

        # 如果缓存的长度不够，或者没有缓存，则重新计算
        if seq_len > self.max_seq_len_cached if self.max_seq_len_cached else True:
            self._update_cos_sin_tables(q, seq_len=max(seq_len, self.max_seq_len_cached or 0, 512))  # 至少缓存到512或当前seq_len
            # 确保缓存的cos/sin与当前seq_len匹配
            if self.cos_cached.shape[0] < seq_len:
                self._update_cos_sin_tables(q, seq_len)  # 再次确保足够长

        # 从缓存中取出当前序列长度所需的部分
        cos_current = self.cos_cached[:seq_len, :]
        sin_current = self.sin_cached[:seq_len, :]

        return apply_rotary_pos_emb(q, k, cos_current, sin_current)


# --- 2. 带有 RoPE 的多头自注意力 (Multi-Head Self-Attention with RoPE) ---
class RoFormerSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)  # 缩放因子

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rotary_pos_emb = RotaryPositionalEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): 输入张量 (batch_size, seq_len, d_model)
            mask (Tensor): 注意力掩码 (batch_size, seq_len)，0表示填充，1表示真实数据。
                           用于防止注意力关注填充部分。
        Returns:
            Tensor: 注意力输出 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. 线性投影 Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (bs, n_heads, seq_len, head_dim)

        # 2. 应用旋转位置编码 (RoPE)
        q, k = self.rotary_pos_emb(q, k)

        # 3. 计算注意力分数
        # (bs, n_heads, seq_len, head_dim) @ (bs, n_heads, head_dim, seq_len) -> (bs, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # 4. 应用注意力掩码
        if mask is not None:
            # mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len)
            # 扩展mask维度以适应注意力分数
            # 填充位置 (mask=0) 对应的注意力分数设为-inf，这样softmax后会接近0
            attn_mask = mask.unsqueeze(1).unsqueeze(1).expand_as(attn_scores)  # (bs, n_heads, seq_len, seq_len)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float("-inf"))

        # 5. Softmax 和 Dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 6. 加权求和 V
        # (bs, n_heads, seq_len, seq_len) @ (bs, n_heads, seq_len, head_dim) -> (bs, n_heads, seq_len, head_dim)
        output = torch.matmul(attn_weights, v)

        # 7. 拼接多头并线性投影
        # (bs, seq_len, num_heads, head_dim) -> (bs, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        return output


# --- 3. 前馈网络 (Feed-Forward Network) ---
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 通常使用 GELU 或 ReLU

    def forward(self, x):
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


# --- 4. RoFormer 编码器层 (RoFormer Encoder Layer) ---
class RoFormerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = RoFormerSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): 输入张量 (batch_size, seq_len, d_model)
            mask (Tensor): 注意力掩码 (batch_size, seq_len)
        Returns:
            Tensor: 输出张量 (batch_size, seq_len, d_model)
        """
        # Pre-LN 结构 (LayerNorm -> Attention -> Add&Dropout -> LayerNorm -> FFN -> Add&Dropout)
        # Sub-layer 1: Multi-Head Self-Attention
        norm_x = self.norm1(x)
        attn_output = self.self_attn(norm_x, mask)
        x = x + self.dropout1(attn_output)  # Add & Dropout

        # Sub-layer 2: Feed-Forward
        norm_x = self.norm2(x)
        ff_output = self.feed_forward(norm_x)
        x = x + self.dropout2(ff_output)  # Add & Dropout

        return x


# --- 5. 完整的 RoFormer 分类器模型 ---
class RoFormer(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout):
        super(RoFormer, self).__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)  # 将原始特征维度映射到d_model
        self.norm = nn.LayerNorm(d_model)

        # 堆叠 RoFormer 编码器层
        self.encoder_layers = nn.ModuleList(
            [
                RoFormerEncoderLayer(d_model, num_heads, d_model * 4, dropout)  # 通常 d_ff 是 d_model 的 2 或 4 倍
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        前向传播。
        Args:
            x (Tensor): 形状为 (batch_size, max_seq_len, input_dim) 的特征张量。
            mask (Tensor): 形状为 (batch_size, max_seq_len) 的注意力掩码，
                           1表示真实数据，0表示填充部分。
        Returns:
            Tensor: 形状为 (batch_size, num_classes) 的分类 logits。
        """
        # 1. 输入特征投影
        x = x.squeeze(1)
        x = self.input_proj(x)  # (batch_size, max_seq_len, d_model)
        x = self.norm(x)

        # 2. 通过所有 RoFormer 编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask)  # (batch_size, max_seq_len, d_model)

        # 3. 序列聚合 (Pooling)
        # 考虑到序列长度不一，使用masked mean pooling是一个常见的选择
        # (batch_size, max_seq_len, d_model) * (batch_size, max_seq_len, 1)
        # 确保 mask 的维度与 x 匹配，以便广播
        masked_output = x * mask.unsqueeze(-1)

        # 计算每个序列的真实长度，避免除以零
        lengths = mask.sum(dim=1).float()  # (batch_size,)

        # 将长度为0的序列的长度设置为1，避免NaN，虽然preprocess.py应该过滤了
        lengths[lengths == 0] = 1.0

        # 对每个序列求平均
        # (batch_size, d_model)
        pooled_output = masked_output.sum(dim=1) / lengths.unsqueeze(-1)

        # 4. Dropout
        pooled_output = self.dropout(pooled_output)

        return pooled_output
