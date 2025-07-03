import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_


class FeedForward(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, activation=nn.GELU, dropout=0.):
        super().__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def window_partition(x: torch.Tensor, window_size: int):
    x = rearrange(x, "b (h_w w1) (w_w w2) c -> (b h_w w_w) w1 w2 c", w1=window_size, w2=window_size)
    return x


def window_reverse(windows, window_size, h, w):
    h_w, w_w = h // window_size,w // window_size
    x = rearrange(windows, "(b h_w w_w) w1 w2 c -> b (h_w w1) (w_w w2) c", h_w=h_w, w_w=w_w)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, heads, qkv_bias=True, attn_dropout=0., proj_dropout=0.):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size) if isinstance(window_size, int) else window_size
        self.heads = heads
        self.qkv_bias = qkv_bias

        self.scale = nn.Parameter(torch.log(10 * torch.ones((heads, 1, 1))), requires_grad=True)
        # relative continuous position bias
        self.cpb_fc = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, heads, bias=False),
        )
        # relative coords table
        self.window_h, self.window_w = self.window_size
        relative_coords_h = torch.arange(-(self.window_h - 1), self.window_h, dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_w - 1), self.window_w, dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h, relative_coords_w])
        ).permute(1, 2, 0).contiguous().unsqueeze(0)    # [1, 2 * window_h - 1, 2 * window_w - 1, 2]
        relative_coords_table[:, :, :, 0] /= self.window_h - 1
        relative_coords_table[:, :, :, 1] /= self.window_w - 1
        relative_coords_table *= 8    # normalize to [-8, 8]
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.) / np.log2(8)
        self.register_buffer('relative_coords_table', relative_coords_table)
        # relative position index
        coord_h = torch.arange(self.window_h)
        coord_w = torch.arange(self.window_w)
        coords = torch.flatten(torch.stack(torch.meshgrid([coord_h, coord_w])), start_dim=1)  # [2, window_h * window_w]
        relative_coords = coords[:, :, None] - coords[:, None, :]    # [2, window_h * window_w, window_h * window_w]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # [window_h * window_w, window_h * window_w, 2]
        relative_coords[:, :, 0] += self.window_h - 1
        relative_coords[:, :, 1] += self.window_w - 1
        relative_coords[:, :, 0] *= 2 * self.window_w - 1
        relative_position_index = relative_coords.sum(-1)    # [window_h * window_w, window_h * window_w]
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.v_bias = nn.Parameter(torch.zeros(dim)) if qkv_bias else None
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        qkv_bias = None if self.qkv_bias is None else \
                   torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # cosine attention
        attention = q @ k.transpose(-2, -1)
        scale = torch.clamp(self.scale, max=torch.log(torch.tensor(1. / 0.01)).to(self.scale.device)).exp()
        attention *= scale

        relative_cpb_table = self.cpb_fc(self.relative_coords_table).view(-1, self.heads)
        relative_cpb = relative_cpb_table[self.relative_position_index.view(-1)].view(
            self.window_h * self.window_w, self.window_h * self.window_w, -1
        ).permute(2, 0, 1).contiguous()
        relative_cpb = 16 * torch.sigmoid(relative_cpb)
        attention += relative_cpb.unsqueeze(0)

        if mask is not None:
            n_w = mask.shape[0]
            attention = attention.view(b // n_w, n_w, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attention = attention.view(-1, self.heads, n, n)
        attention = self.attn_drop(self.softmax(attention))

        x = (attention @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj_drop(self.proj(x))


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, heads, window_size=7, shift_size=0, ff_ratio=4., qkv_bias=True,
                 dropout=0., attn_dropout=0., dropout_path=0., activation=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = (input_resolution, input_resolution) if isinstance(input_resolution, int) else input_resolution
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.ff_ratio = ff_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm(dim)
        self.attention = WindowAttention(dim, window_size, heads, qkv_bias, attn_dropout, proj_dropout=dropout)
        self.drop_path = DropPath(dropout_path) if dropout_path > 0 else nn.Identity()
        self.norm2 = norm(dim)
        self.ff = FeedForward(dim, hidden_channels=int(dim * ff_ratio), activation=activation, dropout=dropout)

        if self.shift_size > 0:
            # SW-MSA
            h, w = self.input_resolution
            mask = torch.zeros((1, h, w, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    mask[:, h_slice, w_slice, :] = count
                    count += 1
            mask_windows = window_partition(mask, self.window_size).view(-1, self.window_size * self.window_size)    # [n_w, window_size, window_size, 1] -> [n_w, window_size * window_size]
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer('attn_mask', attn_mask)

    def forward(self, x):
        h, w = self.input_resolution
        _, n, c = x.shape
        assert n == h * w, f"input feature has wrong size, {n} != {h}*{w}"

        shortcut = x
        x = rearrange(x, 'b (h w) c -> b h w c', h=h, w=w)
        shifted_x = x if self.shift_size == 0 else torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, c)
        # W-MSA/SW-MSA
        attn_windows = self.attention(x_windows, mask=self.attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        # reverse cyclic shift
        x = shifted_x if self.shift_size == 0 else torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = rearrange(x, 'b h w c -> b (h w) c')
        x = shortcut + self.drop_path(self.norm1(x))

        return x + self.drop_path(self.norm2(self.ff(x)))


class PatchMerge(nn.Module):
    def __init__(self, dim, input_resolution, norm=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = (input_resolution, input_resolution) if isinstance(input_resolution, int) else input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm(2 * dim)

    def forward(self, x):
        h, w = self.input_resolution
        b, n, c = x.shape
        assert n == h * w, f"input feature has wrong size, {n} != {h}*{w}"
        assert h % 2 == 0 and w % 2 == 0, f"x size ({h}*{w}) are not even"
        x = rearrange(x, 'b (h w) c -> b h w c', h=h, w=w)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(b, -1, 4 * c)  # [b, h/2 * w/2, 4 * c]

        return self.norm(self.reduction(x))


class BasicLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, heads, window_size, dropout_path, ff_ratio=4., qkv_bias=True,
                 dropout=0., attn_dropout=0., norm=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = (input_resolution, input_resolution) if isinstance(input_resolution, int) else input_resolution
        self.depth = depth
        self.heads = heads

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=self.input_resolution, heads=heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, ff_ratio=ff_ratio,
                                 qkv_bias=qkv_bias, dropout=dropout, attn_dropout=attn_dropout, norm=norm,
                                 dropout_path=dropout_path[i] if isinstance(dropout_path, list) else dropout_path)
            for i in range(depth)
        ])

        self.downsample = downsample(dim=dim, input_resolution=self.input_resolution, norm=norm) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _init_respostnorm(self):
        for block in self.blocks:
            nn.init.constant_(block.norm1.weight, 0)
            nn.init.constant_(block.norm1.bias, 0)
            nn.init.constant_(block.norm2.weight, 0)
            nn.init.constant_(block.norm2.bias, 0)



class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, embed_dim=96, norm=None):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        image_h, image_w = self.image_size
        patch_h, patch_w = self.patch_size
        self.patch_resolution = (image_h // patch_h, image_w // patch_w)
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm(embed_dim) if norm else None

    def forward(self, x):
        _, _, h, w = x.shape
        assert h == self.image_size[0] and w == self.image_size[1], f"input image size ({h}*{w}) doesn't match model ({self.image_size}*{self.image_size})"
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


class SwinTransformer(nn.Module):
    def __init__(self, image_size, patch_size, in_channels=3, num_classes=10, embed_dim=96,
                 depths=None, heads=None, window_size=7, ff_ratio=4., qkv_bias=True, dropout=0.,
                 attn_dropout=0., dropout_path=0.1, norm=nn.LayerNorm, patch_norm=True, ape=False):
        super().__init__()
        if heads is None:
            heads = [3, 6, 12, 24]
        if depths is None:
            depths = [2, 2, 6, 2]
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.ff_ratio = ff_ratio
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embedding = PatchEmbedding(
            image_size, patch_size, in_channels, embed_dim, norm if self.patch_norm else None,
        )
        num_patches = self.patch_embedding.num_patches
        self.patch_resolution = self.patch_embedding.patch_resolution
        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_dropout = nn.Dropout(dropout)
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, dropout_path, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList([])
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i), depth=depths[i], heads=heads[i], window_size=window_size,
                input_resolution=(self.patch_resolution[0] // (2 ** i), self.patch_resolution[1] // (2 ** i)),
                ff_ratio=ff_ratio, qkv_bias=qkv_bias, dropout=dropout, attn_dropout=attn_dropout,
                dropout_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm=norm,
                downsample=PatchMerge if (i < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        self.norm = norm(self.num_features)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for layer in self.layers:
            layer._init_respostnorm()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embedding(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # [b, n, c]
        x = self.avg_pool(x.transpose(1, 2))  # [b, c, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x