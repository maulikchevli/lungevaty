import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from vital.eva import Eva, PatchEmbed
from vital.vit import ViT
from tools.masker import Masker
from einops import rearrange
from torch.nn import MultiheadAttention

class Cumulative_Probability_Layer(nn.Module):
    def __init__(self, num_features, max_followup):
        super(Cumulative_Probability_Layer, self).__init__()

        self.hazard_fc = nn.Linear(num_features, max_followup)
        self.base_hazard_fc = nn.Linear(num_features, 1)
        #self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        mask = torch.ones([max_followup, max_followup])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(
            B, T, T
        )  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        base_hazard = self.base_hazard_fc(x)
        cum_prob = torch.sum(masked_hazards, dim=1) + base_hazard
        return cum_prob
    
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p=0.1):
        """
        A residual block that projects from `dim` to `hidden_dim`, applies a non-linearity
        and dropout, then projects back to `dim`, and adds the original input.
        """
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out + identity
    
    
class FusionLayerWithResidual(nn.Module):
    def __init__(self, input_dim=2304, hidden_dim=1024, output_dim=768, dropout_p=0.1, num_residual_blocks=2):
        """
        Fuses an input vector of dimension `input_dim` through a multi-layer residual network,
        then projects it down to an output vector of dimension `output_dim`.
        """
        super(FusionLayerWithResidual, self).__init__()
        # Optional initial projection to help stabilize training
        self.fc_in = nn.Linear(input_dim, input_dim)
        # Create a sequence of residual blocks
        self.res_blocks = nn.Sequential(*[
            ResidualBlock(input_dim, hidden_dim, dropout_p) 
            for _ in range(num_residual_blocks)
        ])
        # Final projection from input_dim (2304) to output_dim (768)
        self.fc_out = nn.Linear(input_dim, output_dim)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x is assumed to be of shape (batch_size, input_dim)
        x = self.fc_in(x)
        x = self.res_blocks(x)
        x = self.fc_out(x)
        x = self.act(x)
        return x




class Lungevity(nn.Module):
    def __init__(
        self,
        transformer: str = "eva",
        patch_size: int = 16,
        grid_size: list = [10, 10, 10],
        enc_dim: int = 792,
        enc_blocks: int = 12,
        enc_heads: int = 12,
        dropout_rate: float = 0.2,
        num_reg_tokens: int = 0,  # Number of additional tokens for the encoder
        use_cls: bool = True,
        hidden_dim: int = 792,
        max_followup: int = 6,
        fusion_layer: bool = True,
        guided_attention_heads: int = 4,
        use_mean_token: bool = False,
        task: str = "survival",
        num_classes: int = 1,
    ):
        super().__init__()
        
        if transformer == "eva":
            self.encoder =  Eva(
                embed_dim=enc_dim,
                depth=enc_blocks,
                num_heads=enc_heads,
                pos_drop_rate=0.0,
                patch_drop_rate=0.0,
                proj_drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.0,
                ref_feat_shape=grid_size,
                num_reg_tokens=num_reg_tokens,  # Assuming 1 prefix ("cls") token for the encoder
            )

        else:
            self.encoder = ViT(
                embed_dim=enc_dim,
                grid_size=grid_size,
                depth=enc_blocks,
                num_heads=enc_heads,
                drop_rate=dropout_rate
            )

        self.dropout = nn.Dropout(p=dropout_rate)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.pool_token = nn.Parameter(torch.zeros(1, 1, enc_dim))
        self.down_projection = PatchEmbed(patch_size, input_channels=1, embed_dim=enc_dim)

        hidden = hidden_dim
        self.aggregate_fn = lambda x, y, z: x
        if use_cls:
            hidden += hidden_dim
            self.aggregate_fn = lambda x, y, z: torch.concat([x, y], axis=-1)
        if use_mean_token:
            hidden += hidden_dim
            self.aggregate_fn = lambda x, y, z: torch.concat([x, z], axis=-1)
        if use_cls and use_mean_token:
            self.aggregate_fn = lambda x, y, z: torch.concat([x, y, z], axis=-1)

        self.mha = MultiheadAttention(embed_dim=hidden_dim, num_heads=guided_attention_heads, batch_first=True, dropout=dropout_rate)

        if fusion_layer and task == "survival":
            self.classifier = nn.Sequential(*[
                FusionLayerWithResidual(input_dim=hidden, output_dim=hidden_dim, hidden_dim=1024, dropout_p=dropout_rate),
                Cumulative_Probability_Layer(hidden_dim, max_followup)
            ])
        elif fusion_layer and task == "classification":
            self.classifier = nn.Sequential(*[
                FusionLayerWithResidual(input_dim=hidden, output_dim=hidden_dim, hidden_dim=1024, dropout_p=dropout_rate),
                nn.Linear(hidden_dim, num_classes)  # Use configurable num_classes
            ])
        elif not fusion_layer and task == "survival":
            self.classifier = nn.Sequential(*[
                nn.Linear(hidden, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                Cumulative_Probability_Layer(hidden_dim, max_followup)
            ])
        else:
            self.classifier = nn.Sequential(*[
                nn.Linear(hidden, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, num_classes)  # Use configurable num_classes
            ])

        self.grid_size = grid_size

    def attention_pooling(
            self,
            tokens: torch.Tensor
    ):
        B = tokens.shape[0]
        q = self.pool_token.expand(B, -1, -1)  # [B, 1, D]
        k = tokens[:, 1:, :]
        v = tokens[:, 1:, :]

        attns, attn_weights = self.mha(q, k, v, average_attn_weights=False)
        return attns.squeeze(axis=1), attn_weights

    def __call__(
        self,
        input: torch.Tensor,
        return_attention: bool = False,
        return_features: bool = False,
    ):
        x, FD, FW, FH = self.patch_embed(input) #TODO where is cls
        # add cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        if return_features:
             embeddings, enc_attn, enc_features = self.encoder(x, return_attention=return_attention, return_features=True)
        else:
             embeddings, enc_attn = self.encoder(x, return_attention=return_attention)  # (B, num_patches+1, D)

        attn_pooled, attn_weights = self.attention_pooling(embeddings)
        mean_pooled, _ = embeddings[:, 1:, :].max(dim=1)  # shape [B, hidden_dim]
        cls = embeddings[:, 0, :]
        
        pooled_output = self.aggregate_fn(attn_pooled, cls, mean_pooled)
        
        output = self.classifier(pooled_output)
        
        if return_features:
            return output, attn_weights, enc_attn, enc_features, pooled_output

        return output, attn_weights, enc_attn
    
    def initialize_parameters(self):        
        # Initialize (and freeze) pos_embed by sin-cos embedding

        # timm"s trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, "cls_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)
        if hasattr(self, "pool_token"):
            torch.nn.init.normal_(self.cls_token, std=.02)
        if hasattr(self, "mask_token"):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # Initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights) # TODO
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patch_embed(self, x):
        FD, FW, FH = x.shape[2:]  # Full W , ...
        x = self.down_projection(x)
        B, C, D, W, H = x.shape
        num_patches = W * H * D

        x = rearrange(x, "b c d w h -> b (d w h) c")
        return x, FD, FW, FH

    def restore_image(self, x, D, W, H):
        x = rearrange(x, "b (d w h) c -> b c d w h", h=H, w=W, d=D)


    # def forward(self, x):
    #     x, FD, FW, FH = self.patch_embed(x)
    #     x_masked, mask, ids_restore, ids_keep = self.masker(x)
    #     cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
    #     x_masked = torch.cat([cls_token, x_masked], dim=1)

    #     x_masked = self.encoder(x_masked, ids_keep)
    #     x_masked = self.encoding_projection(x_masked)

    #     masked_tokens = self.mask_token.repeat(x_masked.shape[0], ids_restore.shape[1] + 1 - x_masked.shape[1], 1)
    #     all_embeddings = torch.cat((x_masked[:, 1:, :], masked_tokens), dim=1)
    #     all_embeddings = torch.gather(all_embeddings, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_masked.shape[2]))
    #     all_embeddings = torch.cat((x_masked[:, :1, :], all_embeddings), dim=1)

    #     recon_seq = self.decoder(all_embeddings)
    #     recon_seq = self.up_sample(recon_seq)
    #     return recon_seq[:, 1:, :], mask
    

def patchify(im: torch.Tensor, patch_size: list[int, int, int] = [5, 16, 16]):
    """Split image into patches of size patch_size.

    im: [B, S, T, H, W]
    patch_size: a list of 3
    x: [B, L, np.prod(patch_size)] where L = S * T * H * W / np.prod(patch_size)
    """
    assert len(im.shape) == 5
    assert len(patch_size) == 3

    B, S, T, H, W = im.shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = im.reshape(B, S, t, patch_size[0], h, patch_size[1], w, patch_size[2])
    x = torch.einsum("bstphqwr->bsthwpqr", x)
    x = x.reshape(B, S * t * h * w, np.prod(patch_size))
    return x


def unpatchify(x: torch.Tensor, im_shape: list[int], patch_size: list[int, int, int] = [5, 16, 16]):
    """Combine patches into image.

    x: [B, L, np.prod(patch_size) or T * np.prod(patch_size)]
    im_shape: [B, S, T, X, Y]
    im: [B, S, T, X, Y] where X = Y
    """
    assert len(x.shape) == 3
    assert len(patch_size) == 3
    assert len(im_shape) == 5

    B, S, T, H, W = im_shape
    t, h, w = T // patch_size[0], H // patch_size[1], W // patch_size[2]
    x = x.reshape(B, S, t, h, w, patch_size[0], patch_size[1], patch_size[2])
    x = torch.einsum("bsthwpqr->bstphqwr", x)
    im = x.reshape(im_shape)
    return im