import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath, Mlp
from timm.models.vision_transformer import _load_weights
import math
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block

def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv3d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.ReLU(),
            nn.Conv3d(in_dim, dim, 3, 2, 1, bias=False),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x

class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate= 'tanh')
        self.conv2 = nn.Conv3d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm1(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.act1(x)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm2(x)
        x = x.permute(0, 4, 1, 2, 3)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x
    
class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv3d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x
    
class BrainMT(nn.Module):
    def __init__(
            self, 
            img_size=(91, 109, 91), 
            patch_size=4, 
            in_chans=1,
            num_classes=1,
            embed_dim=512, 
            depth=[12, 8], 
            rms_norm=True,
            residual_in_fp32=True,
            fused_add_norm=True,
            bimamba=True,
            num_heads=2,
            qkv_bias=True,
            num_frames=200,
            drop_rate=0.,
            drop_path_rate=0.1,
            attn_drop_rate=0.,
            fc_drop_rate=0.,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            norm_layer=nn.LayerNorm,
            device=None,
            dtype=None,
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.num_features = self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim)

        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=64, dim=128)
        for i in range(2):
            self.add_module(f"conv_block{i}", ConvBlock(dim=int(128 * 2 ** i), drop_path=0., layer_scale=None, kernel_size=3))
            self.add_module(f"downsample{i}", Downsample(dim=int(128 * 2 ** i), keep_dim=False))
        
        # These patch numbers are specific to the conv architecture
        # D, H, W after patch_embed and 2 downsamples
        # Initial: 91, 109, 91
        # After patch_embed (2x stride 2 conv): 91/4, 109/4, 91/4 -> 22, 27, 22
        # After downsample0 (stride 2): 11, 13, 11
        # After downsample1 (stride 2): 5, 6, 5 -> for a 91x109x91 input
        # The original code has 7*6*6, let's keep it for now but this might need adjustment
        num_patches = 7 * 6 * 6

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim, ssm_cfg=ssm_cfg, norm_epsilon=norm_epsilon, rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32, fused_add_norm=fused_add_norm,
                    layer_idx=i, bimamba=bimamba, drop_path=inter_dpr[i], **factory_kwargs,
                )
                for i in range(depth[0])
            ]
        )

        self.drop_path_attn = nn.ModuleList([DropPath(dpr[i]) for i in range(depth[0], sum(depth))])
        self.blocks = nn.ModuleList(
            [
                Attention(
                    embed_dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop_rate,
                    proj_drop=drop_rate, norm_layer=norm_layer,
                ) 
                for i in range(depth[1])
            ]
        )

        self.mlp = Mlp(in_features=embed_dim, hidden_features=int(embed_dim * 2), act_layer=nn.GELU, drop=drop_rate)
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(partial(_init_weights, n_layer=sum(depth),**(initializer_cfg if initializer_cfg is not None else {})))

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)+len(self.blocks)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        B, T, C, D, H, W = x.shape
        x = x.view(B*T, C, D, H, W)
        x = self.patch_embed(x)
        x = self.conv_block0(x)
        x = self.downsample0(x)
        x = self.conv_block1(x)
        x = self.downsample1(x)
        # print(x.shape)
        _, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, D * H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T) # Temporal-first scan mechanism
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        residual = None
        hidden_states = x
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        for idx, (drop_path_attn, block) in enumerate(zip(self.drop_path_attn, self.blocks)):
            hidden_states = hidden_states + drop_path_attn(block(self.norm(hidden_states)))
            hidden_states = hidden_states + drop_path_attn(self.mlp(self.norm(hidden_states)))

        # return only cls token
        return hidden_states[:, 0, :]
        
    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)
        x = self.head(self.head_drop(x))
        return x
