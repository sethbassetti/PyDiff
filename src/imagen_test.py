import torch

from imagen_pytorch import Unet3D

device = torch.device("cuda:0")

unet = Unet3D(
    dim=16,
    text_embed_dim=None,
    num_resnet_blocks=1,
    cond_dim=None,
    num_image_tokens=1,
    dim_mults=(1, 1, 1, 1),
    cond_images_channels=1,
    time_rel_pos_bias_depth=1,
    channels=1,
    cond_on_text=False,
    use_linear_attn=False,
    use_linear_cross_attn=False,
    layer_attns=False,
    attend_at_middle=False,
    attn_dim_head=4,
    attn_heads=4,
    memory_efficient=True,
).to(device)

x = torch.randn(1, 1, 28, 96, 96).to(device)
time = torch.randn(8).to(device)
cond_images = torch.randn(1, 1, 96, 96).to(device)

y = unet(x, time, cond_images=cond_images)

loss = y.mean()
loss.backward()


breakpoint()
