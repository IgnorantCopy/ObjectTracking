import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from flash_attn import flash_attn_func


def positional_embedding_2d(height, width, dim, temperature=10000, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    assert dim % 4 == 0, 'dimension must be divisible by 4'
    dim //= 4
    omega = torch.arange(dim) / (dim - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    return torch.concat((x.sin(), x.cos(), y.sin(), y.cos()), dim=-1).type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        proj_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.dropout = dropout
        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=False),
            nn.Dropout(dropout)
        ) if proj_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        x = flash_attn_func(q, k, v, self.dropout, self.scale)
        x = rearrange(x, 'b h n d -> b n (h d)')
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout),
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class InnerViT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, image_size, patch_size, in_channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        height, width = self.image_size
        patch_height, patch_width = self.patch_size
        patch_dim = in_channels * patch_height * patch_width
        assert height % patch_height == 0 and width % patch_width == 0, 'image size must be divisible by patch_size'

        self.patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, (height // patch_height) * (width // patch_width) + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, x):
        x = self.patch_embedding(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        return x.unsqueeze(1)


class OuterVit(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, image_size, patch_size, in_channels=1,
                 num_classes=10, seq_len=180, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.inner_patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        image_height, image_width = self.image_size
        inner_patch_height, inner_patch_width = patch_size
        height, width = (image_height // inner_patch_height) * (image_width // inner_patch_width) + 1, dim
        outer_patch_dim = 1 * height * width

        self.inner_vit = InnerViT(dim, depth, heads, mlp_dim, image_size, patch_size, in_channels, dim_head, dropout, emb_dropout)
        self.patch_embedding = nn.Sequential(
            Rearrange('(b s) c n d -> b s (c n d)', s=seq_len),
            nn.LayerNorm(outer_patch_dim),
            nn.Linear(outer_patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.to_latent = nn.Identity()
        self.cls_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = rearrange(x, "b s c h w -> (b s) c h w")
        x = self.inner_vit(x)   # [b * s, 1, n, dim]
        x = self.patch_embedding(x)     # [b, s, dim]
        b, s, _ = x.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(s + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.to_latent(x)
        return self.cls_head(x)
