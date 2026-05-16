# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.scale_block import ScaleBlock


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim) for _ in range(num_upscale)],
        )

    def _predict(self, x: torch.Tensor):
        q = x[:, : self.num_q, :]

        class_logits = self.class_head(q)

        x = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        x = x.transpose(1, 2).reshape(
            x.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )

        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x)
        )

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        B, N, C = x.shape

        if rope is not None:
            # When queries are present, the sequence is [queries, prefix_tokens, patch_tokens].
            # RoPE must only be applied to patch_tokens (and prefix_tokens skip it).
            # EvaAttention applies RoPE to tokens after num_prefix_tokens, but with queries
            # inserted before prefix, the indices are wrong and rope is too small.
            # Handle this manually: split queries from prefix+patches.
            num_prefix = self.encoder.backbone.num_prefix_tokens
            num_patches_expected = self.encoder.backbone.patch_embed.grid_size[0] * self.encoder.backbone.patch_embed.grid_size[1]

            if N > num_prefix + num_patches_expected:
                # Queries are present — split and handle RoPE manually
                queries = x[:, :self.num_q, :]
                prefix_and_patches = x[:, self.num_q:, :]

                # Apply attention to prefix+patches with RoPE
                head_dim = module.qkv.weight.shape[0] // 3 // module.num_heads
                qkv = module.qkv(prefix_and_patches).reshape(B, N - self.num_q, 3, module.num_heads, head_dim)
                q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
                q, k = module.q_norm(q), module.k_norm(k)

                # Apply RoPE to q, k (only to patch tokens, skip prefix)
                from timm.layers.pos_embed_sincos import apply_rot_embed_cat
                half = getattr(module, 'rotate_half', False)
                npt = module.num_prefix_tokens
                q = torch.cat([q[:, :, :npt, :], apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)
                k = torch.cat([k[:, :, :npt, :], apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half)], dim=2).type_as(v)

                # Now handle queries separately (no RoPE)
                qkv_q = module.qkv(queries).reshape(B, self.num_q, 3, module.num_heads, head_dim)
                q_q, k_q, v_q = qkv_q.permute(2, 0, 3, 1, 4).unbind(0)
                q_q, k_q = module.q_norm(q_q), module.k_norm(k_q)

                # Concatenate: queries (no RoPE) + prefix+patches (with RoPE)
                q = torch.cat([q_q, q], dim=2)
                k = torch.cat([k_q, k], dim=2)
                v = torch.cat([v_q, v], dim=2)

                if mask is not None:
                    mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

                dropout_p = module.attn_drop.p if self.training else 0.0

                if module.fused_attn:
                    x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
                else:
                    attn = (q @ k.transpose(-2, -1)) * module.scale
                    if mask is not None:
                        attn = attn.masked_fill(~mask, float("-inf"))
                    attn = F.softmax(attn, dim=-1)
                    attn = module.attn_drop(attn)
                    x = attn @ v

                x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))
                return x

            # No queries present — delegate to module directly
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, rope, mask)[0]

        head_dim = module.qkv.weight.shape[0] // 3 // module.num_heads
        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.encoder.backbone.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
        )
        return attn_mask

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            pos_out = self.encoder.backbone._pos_embed(x)
            if isinstance(pos_out, tuple):
                x, rope = pos_out
            else:
                x = pos_out

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

        mask_logits, class_logits = self._predict(self.encoder.backbone.norm(x))
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
        )
