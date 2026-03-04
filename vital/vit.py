import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import Block

class ViT(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    Modified from timm implementation
    """
    def __init__(self, embed_dim=384, grid_size=[10, 10, 10], depth=4,
                 num_heads=6, mlp_ratio=4., qkv_bias=True, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., 
                 norm_layer=None, act_layer=None):
        super().__init__()

        self.pos_embed = build_3d_sincos_position_embedding(grid_size, embed_dim, add_cls=False)

        self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                proj_drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
     


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    def _pos_embed(self, x, indices=None):
        pos_embed = self.pos_embed.repeat(x.shape[0], 1, 1)
        if indices is not None:
            pos_embed = torch.gather(pos_embed, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, self.pos_embed.shape[-1]))
        cls = x[:, :1, :]
        remaining = x[:, 1:, :] + pos_embed
        x = torch.cat((cls, remaining), dim=1)
        x = self.pos_drop(x)
        return x

    def forward(self, x, indices=None):
        x = self._pos_embed(x, indices)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x
    
def build_3d_sincos_position_embedding(grid_size, embed_dim, add_cls=True, temperature=10000.):
    h, w, d = grid_size
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_d = torch.arange(d, dtype=torch.float32)

    grid_h, grid_w, grid_d = torch.meshgrid(grid_h, grid_w, grid_d)

    assert embed_dim % 6 == 0, 'Embed dimension must be divisible by 6 for 3D sin-cos position embedding'
    pos_dim = embed_dim // 6

    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_d = torch.einsum('m,d->md', [grid_d.flatten(), omega])
    pos_emb = torch.cat([torch.sin(out_h), torch.cos(out_h), torch.sin(out_w), torch.cos(out_w), torch.sin(out_d), torch.cos(out_d)], dim=1)[None, :, :]

    if add_cls:
        pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
        pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
    else:
        pos_embed = nn.Parameter(pos_emb)
    pos_embed.requires_grad = False
    return pos_embed


