
from typing import Tuple, Type, Sequence
import torch
from torch import nn

import math
from typing import Tuple, Callable, Optional

import numpy as np
import torch
from timm.layers import (
    PatchDropout,
    trunc_normal_,
    apply_keep_indices_nlc,
    RotaryEmbeddingCat,
)
from vital.timm_eva import EvaBlock
from torch import nn
from torch.nn import LayerNorm
from torch.utils.checkpoint import checkpoint

from timm.layers import RotaryEmbeddingCat
from einops import rearrange

class AbstractDynamicNetworkArchitectures(nn.Module):

    def __init__(self):
        super(AbstractDynamicNetworkArchitectures, self).__init__()
        # Key to the position holding all the encoder weights
        self.key_to_encoder: str
        # Key to the full stem -- Can be located within or outside the encoder
        self.key_to_stem: str
        # Not sure yet if we need anything but this -- but minor redundancy is okay I suppose
        # Key to the weights that are dependent on the input channels.
        #   Can hold multiple weights (e.g. for bad weight mappings like in this repo >.<' )
        self.keys_to_in_proj: Sequence[str]
        self.key_to_lpe: str | None = None  # LPE == Learnable Positional Embedding


_PRIMUS_CONFIGS = {
    "S": {
        "eva_depth": 12,
        "eva_numheads": 6,
        "embed_dim": 396,
    },
    "B": {
        "eva_depth": 12,
        "eva_numheads": 12,
        "embed_dim": 792,
    },
    "M": {
        "eva_depth": 16,
        "eva_numheads": 12,
        "embed_dim": 864,
    },
    "L": {
        "eva_depth": 24,
        "eva_numheads": 16,
        "embed_dim": 1056,
    },
}


class LayerNormNd(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        idx = (None, slice(None), *([None] * (x.ndim - 2)))
        x = (
            self.weight[idx] * x
            + self.bias[idx]
        )
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    Loosely inspired by https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py#L364

    """

    def __init__(
        self,
        patch_size: Tuple[int, ...] = (16, 16, 16),
        input_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            patch_size (Tuple): patch size.
            padding (Tuple): padding size of the projection layer.
            input_channels (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = convert_dim_to_conv_op(len(patch_size))(
            input_channels, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        returns shape (B, embed_dim, px, py, pz) where (px, py, pz) is patch_size.
        This output will need to be rearranged to whatever your transformer expects!
        """
        x = self.proj(x)
        return x


class PatchDecode(nn.Module):
    """
    Loosely inspired by SAM decoder
    https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py#L53
    """

    def __init__(
        self,
        patch_size,
        embed_dim: int,
        out_channels: int,
        norm=LayerNormNd,
        activation=nn.GELU,
    ):
        """
        patch size must be 2^x, so 2, 4, 8, 16, 32, etc. Otherwise we die
        """
        super().__init__()

        def _round_to_8(inp):
            return int(max(8, np.round((inp + 1e-6) / 8) * 8))

        num_stages = int(np.log(max(patch_size)) / np.log(2))
        strides = [[2 if (p / 2**n) % 2 == 0 else 1 for p in patch_size] for n in range(num_stages)][::-1]
        dim_red = (embed_dim / (2 * out_channels)) ** (1 / num_stages)

        # don't question me
        channels = [embed_dim] + [_round_to_8(embed_dim / dim_red ** (x + 1)) for x in range(num_stages)]
        channels[-1] = out_channels

        stages = []
        for s in range(num_stages - 1):
            stages.append(
                nn.Sequential(
                    nn.ConvTranspose3d(channels[s], channels[s + 1], kernel_size=strides[s], stride=strides[s]),
                    norm(channels[s + 1]),
                    activation(),
                )
            )
        stages.append(nn.ConvTranspose3d(channels[-2], channels[-1], kernel_size=strides[-1], stride=strides[-1]))
        self.decode = nn.Sequential(*stages)

    def forward(self, x):
        """
        Expects input of shape (B, embed_dim, px, py, pz)! This will require you to reshape the output of your transformer!
        """
        return self.decode(x)
    
def convert_dim_to_conv_op(dimension: int):
    """
    :param dimension: 1, 2 or 3
    :return: conv Class of corresponding dimension
    """
    if dimension == 1:
        return nn.Conv1d
    elif dimension == 2:
        return nn.Conv2d
    elif dimension == 3:
        return nn.Conv3d
    else:
        raise ValueError("Unknown dimension. Only 1, 2 and 3 are supported")


class InitWeights_He(object):
    def __init__(self, neg_slope: float = 1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class InitWeights_XavierUniform(object):
    def __init__(self, gain: int = 1):
        self.gain = gain

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.xavier_uniform_(module.weight, self.gain)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class Eva(nn.Module):
    """Eva Vision Transformer w/ Abs & Rotary Pos Embed

    This class implements the EVA and EVA02 models that were based on the BEiT ViT variant
      * EVA - abs pos embed, global avg pool
      * EVA02 - abs + rope pos embed, global avg pool, SwiGLU, scale Norm in MLP (ala normformer)


    """

    def __init__(
        self,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        qkv_bias: bool = True,
        qkv_fused: bool = False,
        mlp_ratio: float = 4 * 2 / 3,
        swiglu_mlp: bool = True,
        scale_mlp: bool = True,
        scale_attn_inner: bool = False,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        drop_path_rate: float = 0.0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        norm_layer: Callable = LayerNorm,
        init_values: Optional[float] = None,
        use_abs_pos_emb: bool = True,
        use_rot_pos_emb: bool = True,
        dynamic_img_size: bool = False,
        ref_feat_shape: Optional[Tuple[int, ...]] = None,  # 224/14
        num_reg_tokens: int = 0,
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        block_fn=EvaBlock,
    ):
        """
        Diff to timm implementation

        - removed patch embedding, we expect embeded patches
        - removed classification token, we use features at the end
        - removed head
        - dynamic image size is not supported, but left in for future stuff
        - self.cls_token removed
        - removed postnorm block support
        """
        super().__init__()
        if rope_kwargs is None:
            rope_kwargs = {}

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        self.num_prefix_tokens = num_reg_tokens

        num_patches = np.prod(ref_feat_shape, dtype=int)

        self.pos_embed = (
            nn.Parameter(torch.zeros(1, num_patches + self.num_prefix_tokens, embed_dim)) if use_abs_pos_emb else None
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            if len(ref_feat_shape) == 3:
                rope_dim = round(embed_dim // num_heads / 1.5)
                assert rope_dim == embed_dim / num_heads / 1.5, "rope dim must be divsible by (num_heads * 1.5)"
                assert rope_dim % 4 == 0, "rope dim must be divisible by 4"
            else:
                rope_dim = embed_dim // num_heads
            self.rope = rope_impl(
                rope_dim, in_pixels=False, feat_shape=ref_feat_shape, ref_feat_shape=ref_feat_shape, **rope_kwargs
            )
        else:
            self.rope = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        block_fn = block_fn
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_fused=qkv_fused,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=swiglu_mlp,
                    scale_mlp=scale_mlp,
                    scale_attn_inner=scale_attn_inner,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    num_prefix_tokens=self.num_prefix_tokens,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = {"pos_embed", "cls_token"}
        return nwd

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )
        return matcher

    def _pos_embed(self, x, indices=None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.dynamic_img_size:
            raise NotImplementedError("dynamic_img_size is not implemented at the moment")
            B, H, W, C = x.shape
            if self.pos_embed is not None:
                pos_embed = resample_abs_pos_embed_3d(
                    self.pos_embed,
                    (H, W),
                    num_prefix_tokens=self.num_prefix_tokens,
                )
            else:
                pos_embed = None
            x = x.view(B, -1, C)
            rot_pos_embed = self.rope.get_embed(shape=(H, W)) if self.rope is not None else None
        else:
            pos_embed = self.pos_embed.repeat(x.shape[0], 1, 1)

            rot_pos_embed = self.rope.get_embed() if self.rope is not None else None
            rot_pos_embed = rot_pos_embed.repeat(x.shape[0], 1, 1)

        if pos_embed is not None:
            if indices is not None:
                pos_embed = torch.gather(pos_embed, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, pos_embed.shape[-1]))
            cls = x[:, :1, :]
            remaining = x[:, 1:, :] + pos_embed
            x = torch.cat((cls, remaining), dim=1)
        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        # if self.patch_drop is not None:
        #     x, keep_indices = self.patch_drop(x)
        #     if rot_pos_embed is not None and keep_indices is not None:
        #         rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        #     return x, rot_pos_embed, keep_indices
        # else:
        #     return x, rot_pos_embed, None

        if indices is not None:
            rot_pos_embed = torch.gather(rot_pos_embed, dim=1, index=indices.unsqueeze(-1).repeat(1, 1, rot_pos_embed.shape[-1]))
        cls_rot_pos_embed = torch.ones((rot_pos_embed.shape[0], 1, rot_pos_embed.shape[-1]), device=rot_pos_embed.device, dtype=rot_pos_embed.dtype)
        cls_rot_pos_embed[:, :, rot_pos_embed.shape[-1] // 2:] = 0.0
        rot_pos_embed = torch.cat((cls_rot_pos_embed, rot_pos_embed), dim=1)

        return x, rot_pos_embed, None

    def forward_features(self, x, indices, return_attention=False, return_features=False):
        x, rot_pos_embed, keep_indices = self._pos_embed(x, indices)
        attn_weights = []
        features = []
        for blk in self.blocks:
            if not return_attention and self.grad_checkpointing and not torch.jit.is_scripting():
                x, attn_wts = checkpoint(blk, x, rope=rot_pos_embed)
            else:
                x, attn_wts = blk(x, rope=rot_pos_embed, return_attention=return_attention)
            attn_weights.append(attn_wts)
            if return_features:
                features.append(x)
                
        x = self.norm(x)
        
        result = [x, keep_indices]
        if return_attention:
            result.append(attn_weights)
        else:
            result.append(None)
            
        if return_features:
            result.append(features)
            
        return tuple(result)

    def forward(self, x, indices=None, return_attention=False, return_features=False):
        ret = self.forward_features(x, indices, return_attention=return_attention, return_features=return_features)
        
        # ret structure: (x, keep_indices, attn_weights, [opt]features)
        embeddings = ret[0]
        # keep_indices = ret[1] # unused here
        attn_weights = ret[2]
        
        if return_features:
            return embeddings, attn_weights, ret[3]
            
        return embeddings, attn_weights


class Primus(AbstractDynamicNetworkArchitectures):

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        num_classes: int,
        eva_depth: int = 24,
        eva_numheads: int = 16,
        input_shape: Tuple[int, ...] = None,
        decoder_norm=LayerNormNd,
        decoder_act=nn.GELU,
        num_register_tokens: int = 0,
        use_rot_pos_emb: bool = True,
        use_abs_pos_embed: bool = True,
        mlp_ratio=4 * 2 / 3,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        proj_drop_rate: float = 0.0,  # drops out things related to the projection. That is in the MLP and at the end of EVA attention
        attn_drop_rate: float = 0.0,  # drops attention, meaning connections between patches may bebroken up at random
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
    ):
        """
        Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        assert input_shape is not None
        assert len(input_shape) == 3, "Currently on ly 3d is supported"
        assert all([j % i == 0 for i, j in zip(patch_embed_size, input_shape)])

        super().__init__()
        self.key_to_encoder = "eva"
        self.key_to_stem = "down_projection"
        self.keys_to_in_proj = ("down_projection.proj",)
        self.key_to_lpe = "eva.pos_embed"

        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)
        self.up_projection = PatchDecode(
            patch_embed_size, embed_dim, num_classes, norm=decoder_norm, activation=decoder_act
        )

        # we need to compute the ref_feat_shape for eva
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=tuple([i // ds for i, ds in zip(input_shape, patch_embed_size)]),
            num_reg_tokens=num_register_tokens,
            use_rot_pos_emb=use_rot_pos_emb,
            use_abs_pos_emb=use_abs_pos_embed,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )
        # self.mask_token =
        self.mask_token: torch.Tensor
        self.register_buffer("mask_token", torch.zeros(1, 1, embed_dim))

        if num_register_tokens > 0:
            self.register_tokens = (
                nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
            )
            nn.init.normal_(self.register_tokens, std=1e-6)
        else:
            self.register_tokens = None

        self.down_projection.apply(InitWeights_He(1e-2))
        self.up_projection.apply(InitWeights_He(1e-2))
        # eva has its own initialization

    def restore_full_sequence(self, x, keep_indices, num_patches):
        """
        Restore the full sequence by filling blanks with mask tokens and reordering.
        """
        if keep_indices is None:
            return x, None
        B, num_kept, C = x.shape
        device = x.device

        # Create mask tokens for missing patches
        num_masked = num_patches - num_kept
        mask_tokens = self.mask_token.repeat(B, num_masked, 1)

        # Prepare an empty tensor for the restored sequence
        restored = torch.zeros(B, num_patches, C, device=device)
        restored_mask = torch.zeros(B, num_patches, dtype=torch.bool, device=device)

        # Assign the kept patches and mask tokens in the correct positions
        for i in range(B):
            kept_pos = keep_indices[i]
            # masked_pos_prior = torch.tensor([j for j in range(num_patches) if j not in kept_pos], device=device)
            # replacement of list comprehension
            # kept_pos_tensor = torch.tensor(kept_pos, device=device)  # Ensure kept_pos is a tensor
            all_indices = torch.arange(num_patches, device=device)  # Create tensor of all indices
            mask = torch.ones(num_patches, device=device, dtype=torch.bool)  # Start with all True
            mask[kept_pos] = False  # Set kept positions to False
            masked_pos = all_indices[mask]  # Extract indices not in kept_pos

            restored[i, kept_pos] = x[i]
            restored[i, masked_pos] = mask_tokens[i, : len(masked_pos)]
            restored_mask[i, kept_pos] = True

        return (restored, restored_mask)

    def forward(self, x, ret_mask=False):
        FW, FH, FD = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        # last output of the encoder is the input to EVA
        B, C, W, H, D = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c w h d -> b (w h d) c")
        if self.register_tokens is not None:
            x = torch.cat(
                (
                    self.register_tokens.expand(x.shape[0], -1, -1),
                    x,
                ),
                dim=1,
            )
        x, keep_indices = self.eva(x)

        if self.register_tokens is not None:
            x = x[:, self.register_tokens.shape[1] :]  # Removes the register tokens
        # In-fill in-active patches with empty tokens
        restored_x, restoration_mask = self.restore_full_sequence(x, keep_indices, num_patches)
        x = rearrange(restored_x, "b (w h d) c -> b c w h d", h=H, w=W, d=D)
        if restoration_mask is not None:
            mask = rearrange(restoration_mask, "b (w h d) -> b w h d", h=H, w=W, d=D)
            full_mask = (
                mask.repeat_interleave(FW // W, dim=1)
                .repeat_interleave(FH // H, dim=2)
                .repeat_interleave(FD // D, dim=3)
            )
            full_mask = full_mask[:, None, ...]  # Add channel dimension  # [B, 1, W, H, D]
        else:
            full_mask = None

        dec_out = self.up_projection(x)
        if ret_mask:
            return dec_out, full_mask
        else:
            return dec_out

    def compute_conv_feature_map_size(self, input_size):
        raise NotImplementedError("yuck")


class PrimusX(Primus):

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        config_name: str,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=None,
        scale_attn_inner=False,
    ):
        conf = _PRIMUS_CONFIGS[config_name]
        super().__init__(
            input_channels=input_channels,
            embed_dim=conf["embed_dim"],
            patch_embed_size=patch_embed_size,
            num_classes=output_channels,
            eva_depth=conf["eva_depth"],
            eva_numheads=conf["eva_numheads"],
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusS(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-S Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="S",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusB(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-B Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="B",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusM(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-M Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="M",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )


class PrimusL(PrimusX):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        patch_embed_size: Tuple[int, ...],
        input_shape: Tuple[int, ...] = None,
        drop_path_rate=0,  # drops computations (multihead attention, mlp), Implementation of scaling might be useless here because this is not batch normed
        patch_drop_rate: float = 0.0,  # drops input patches, may be used for MAE style pretraining
        rope_impl=RotaryEmbeddingCat,
        rope_kwargs=None,
        init_values=0.1,
        scale_attn_inner=True,
    ):
        """
        Official Primus-L Architecture as proposed in the Primus paper (https://arxiv.org/pdf/2503.01835)
        `Primus: Enforcing Attention Usage for 3D Medical Image Segmentation`

        consists of simple patch_embedding, a EVA ViT encoder with a few adatptations and a simple patch decoder.
        """
        super().__init__(
            input_channels=input_channels,
            output_channels=output_channels,
            config_name="L",
            patch_embed_size=patch_embed_size,
            input_shape=input_shape,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=patch_drop_rate,
            rope_impl=rope_impl,
            rope_kwargs=rope_kwargs,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )
