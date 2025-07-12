import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from flash_attn import flash_attn_func


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


class ViViT(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, image_size, patch_size, in_channels=1,
                 num_classes=10, seq_len=180, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size

        height, width = self.image_size
        patch_height, patch_width = self.patch_size
        patch_dim = in_channels * patch_height * patch_width
        assert height % patch_height == 0 and width % patch_width == 0, 'image size must be divisible by patch_size'

        self.patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        num_patches = (height // patch_height) * (width // patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, num_patches + 1, dim))
        self.spatial_token = nn.Parameter(torch.randn(1, 1, dim))
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.dropout = nn.Dropout(emb_dropout)

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.patch_embedding(x)
        b, t, n, _ = x.shape
        spatial_tokens = repeat(self.spatial_token, '() n d -> b t n d', b=b, t=t)
        x = torch.cat((spatial_tokens, x), dim=2)
        x += self.pos_embedding[:, :, :(n + 1)]
        x = self.dropout(x)

        x = rearrange(x, 'b t n d -> (b t) n d')
        x = self.spatial_transformer(x)
        x = rearrange(x[:, 0], '(b t) ... -> b t ...', b=b)

        temporal_tokens = repeat(self.temporal_token, '() n d -> b n d', b=b)
        x = torch.cat((temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)
        x = x.mean(dim=1)
        return self.head(x)
