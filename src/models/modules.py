import torch
from torch import nn
from einops import rearrange


def default(val, default_val):
    """Return default_val if val is None."""
    return default_val if val is None else val


def Normalize(dim_in, num_groups=32):
    return nn.GroupNorm(num_groups, dim_in, eps=1e-6, affine=True)

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0,1,0,1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(
        self, dim_in, dim_out=None, time_dim=512, dropout=0.0, conv_shortcut=False
    ):
        super().__init__()

        dim_out = default(dim_out, dim_in)
        self.act = nn.SiLU()

        # First conv block
        self.norm1 = Normalize(dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1)

        if time_dim > 0:
            self.time_proj = nn.Linear(time_dim, dim_out)

        # Second conv block
        self.norm2 = Normalize(dim_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1)

        # Determine how to handle the residual connection
        if dim_in != dim_out and conv_shortcut:
            self.res_conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=3, stride=1, padding=1
            )

        elif dim_in != dim_out and not conv_shortcut:
            self.res_conv = nn.Conv2d(
                dim_in, dim_out, kernel_size=1, stride=1, padding=0
            )

        else:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb=None):
        # First block
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        # Add time embeddings to representation
        if time_emb is not None:
            h += self.time_proj(self.act(time_emb))[:, :, None, None]

        # Second block
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Residual connection
        h += self.res_conv(x)

        return h


class AttnBlock(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.norm = Normalize(in_dim)
        self.q_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.k_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)
        self.v_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

        self.out_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = self.norm(x)

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        _, channels, height, width = h.shape

        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        v = rearrange(v, "b c h w -> b c (h w)")

        # Compute attn matrix
        weights = torch.bmm(q, k)
        weights *= channels**0.5
        weights = nn.functional.softmax(weights, dim=-1)

        # Attend to values

        # Switch the last two dimensions
        weights = rearrange(weights, "b q k -> b k q")
        h = torch.bmm(v, weights)

        # Expand back to the height and width dimensions
        h = rearrange(h, "b c (h w) -> b c h w", h=height, w=width)
        h = self.out_proj(h)

        # Residual connection
        return x + h

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""
    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)

def make_attn(in_channels, attn_type="vanilla"):
    assert attn_type in ["vanilla", "linear", "none"], f'attn_type {attn_type} unknown'
    if attn_type == "vanilla":
        return AttnBlock(in_channels)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


if __name__ == "__main__":
    x = torch.randn(16, 64, 32, 32)
    t = torch.randn(16, 512)

    block = ResnetBlock(
        dim_in=64, dim_out=64, time_dim=512, dropout=0.0, conv_shortcut=False
    )

    out = block(x, t)
