import torch
from torch import nn

from modules import ResnetBlock, make_attn, Downsample, Upsample, Normalize


class Encoder(nn.Module):
    def __init__(
        self,
        dim_in,
        model_dim,
        z_dim,
        num_res_blocks,
        spatial_resolution,
        attn_resolutions,
        dim_mults=(1, 2, 4, 8),
        dropout=0.0,
        resamp_with_conv=True,
        attn_type="vanilla",
        double_z=True,
    ):
        super().__init__()

        self.dim_mults = dim_mults
        self.num_res_blocks = num_res_blocks
        self.num_resolutions = len(dim_mults)

        self.in_conv = nn.Conv2d(dim_in, model_dim, kernel_size=3, stride=1, padding=1)

        # Incoming dimensions for each level
        in_dim_mults = (1,) + tuple(dim_mults)
        current_resolution = spatial_resolution

        self.down_blocks = nn.ModuleList()

        # Iterate through each level
        for level, dim_mult in enumerate(self.dim_mults):
            block_dim_in = model_dim * in_dim_mults[level]
            block_dim_out = model_dim * dim_mult

            sub_blocks = nn.ModuleList()

            # Construct the sub-residual/attn blocks for each level
            for _ in range(num_res_blocks):
                sub_block = nn.Module()
                sub_block.resblock = ResnetBlock(
                    block_dim_in, block_dim_out, time_dim=0, dropout=dropout
                )

                # Add either an attention block or an identity block
                if current_resolution in attn_resolutions:
                    sub_block.attn = make_attn(block_dim_out, attn_type)
                else:
                    sub_block.attn = nn.Identity()

                sub_blocks.append(sub_block)

                block_dim_in = block_dim_out

            # This represents all the layers for a single depth level
            level_block = nn.Module()
            level_block.sub_blocks = sub_blocks

            # Decrease spatial res on all but last block
            if level != (self.num_resolutions - 1):
                level_block.downsample = Downsample(block_dim_in, resamp_with_conv)
                current_resolution //= 2
            else:
                level_block.downsample = nn.Identity()

            self.down_blocks.append(level_block)

        self.mid = nn.Sequential(
            ResnetBlock(block_dim_in, block_dim_in, time_dim=0, dropout=dropout),  # type: ignore
            make_attn(block_dim_in, attn_type),  # type: ignore
            ResnetBlock(block_dim_in, block_dim_in, time_dim=0, dropout=dropout),  # type: ignore
        )

        self.out_norm = Normalize(block_dim_in)  # type: ignore
        self.act = nn.SiLU()
        self.out_conv = nn.Conv2d(
            block_dim_in, 2 * z_dim if double_z else z_dim, kernel_size=3, stride=1, padding=1  # type: ignore
        )

    def forward(self, x):
        h = self.in_conv(x)

        for level_block in self.down_blocks:
            # Perform a series of resblocks and attn layers
            for sub_block in level_block.sub_blocks:  # type: ignore
                h = sub_block.resblock(h)
                h = sub_block.attn(h)

            # Finally, downsample the spatial resolution
            h = level_block.downsample(h)  # type: ignore

        h = self.mid(h)
        h = self.out_norm(h)
        h = self.act(h)
        h = self.out_conv(h)

        return h
    

class Decoder(nn.Module):
    def __init__(self, *, model_dim, dim_out, dim_mults=(1,2,4,8), num_res_blocks,
                 attn_resolutions, dropout=0.0, resamp_with_conv=True,
                 spatial_resolution, z_dim, give_pre_end=False, tanh_out=False, use_linear_attn=False,
                 attn_type="vanilla", **ignorekwargs):
        super().__init__()
        if use_linear_attn: attn_type = "linear"
        self.ch = model_dim
        self.temb_ch = 0
        self.num_resolutions = len(dim_mults)
        self.num_res_blocks = num_res_blocks
        self.resolution = spatial_resolution
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,)+tuple(dim_mults)
        block_in = model_dim*dim_mults[self.num_resolutions-1]
        curr_res = spatial_resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_dim,curr_res,curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_dim,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(dim_in=block_in,
                                       dim_out=block_in,
                                       time_dim=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(dim_in=block_in,
                                       dim_out=block_in,
                                       time_dim=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = model_dim*dim_mults[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(dim_in=block_in,
                                         dim_out=block_out,
                                         time_dim=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.act = nn.SiLU()
        self.conv_out = torch.nn.Conv2d(block_in,
                                        dim_out,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)   # type: ignore
        h = self.mid.attn_1(h)        # type: ignore
        h = self.mid.block_2(h, temb)   # type: ignore

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)    # type: ignore
                if len(self.up[i_level].attn) > 0:  # type: ignore
                    h = self.up[i_level].attn[i_block](h)   # type: ignore
            if i_level != 0:
                h = self.up[i_level].upsample(h)    # type: ignore

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)
        if self.tanh_out:
            h = torch.tanh(h)
        return h


if __name__ == "__main__":
    encoder = Encoder(
        dim_in=3,
        model_dim=64,
        z_dim=512,
        num_res_blocks=2,
        spatial_resolution=32,
        attn_resolutions=[16],
        dim_mults=(1, 2, 4, 8),
    )

    decoder = Decoder(
        model_dim=64,
        dim_out=3,
        dim_mults=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=[16],
        z_dim=1024,
        spatial_resolution=32,
    )

    x = torch.randn(1, 3, 32, 32)

    out = encoder(x)

    new_out = decoder(out)
    breakpoint()
