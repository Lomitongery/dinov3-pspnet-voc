import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm for 2D feature maps (channel-only normalization)."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__(num_channels, eps=eps)

    def forward(self, x):
        # x: (B, C, H, W) → permute to (B, H, W, C) → LayerNorm → permute back
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x


class ScaleBlock(nn.Module):
    """Upscales feature map 2x using transposed conv + depthwise conv + LayerNorm."""

    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=1,
            groups=embed_dim, bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x


class EoMT(nn.Module):
    """Encoder-only Mask Transformer for semantic segmentation."""

    def __init__(self, num_classes=21, num_q=100, num_blocks=4,
                 pretrained=True, masked_attn_enabled=True):
        super().__init__()
        self.num_classes = num_classes
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        # ── Backbone: DINOv3 ViT-S/16 ──────────────────────────────────
        self.backbone = timm.create_model(
            'vit_small_patch16_dinov3.lvd1689m',
            pretrained=pretrained,
            num_classes=0,
        )
        # Freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = self.backbone.embed_dim  # 384
        patch_size = self.backbone.patch_embed.patch_size  # (16, 16)
        self.patch_size = patch_size

        # Number of prefix tokens (cls + registers)
        self.num_prefix_tokens = getattr(
            self.backbone, 'num_prefix_tokens',
            1 + 4,  # fallback: 1 cls + 4 register tokens for DINOv3
        )

        # ── Learnable queries ──────────────────────────────────────────
        self.q = nn.Embedding(num_q, embed_dim)

        # ── Prediction heads ───────────────────────────────────────────
        self.class_head = nn.Linear(embed_dim, num_classes + 1)  # +1 for "no object"

        self.mask_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── Upscale blocks ─────────────────────────────────────────────
        num_upscale = max(1, int(math.log2(max(patch_size))) - 2)
        # patch_size=16 → max(1, 4-2) = 2 ScaleBlocks (16→32→64)
        # patch_size=14 → max(1, ~3.8-2) = 1 ScaleBlock  (14→28)
        self.upscale = nn.Sequential(
            *[ScaleBlock(embed_dim) for _ in range(num_upscale)]
        )

        # ── Annealing buffer ───────────────────────────────────────────
        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

    # ── Forward helpers ────────────────────────────────────────────────

    def _get_grid_size(self, num_patch_tokens):
        """Compute spatial grid size from number of patch tokens."""
        # Assume square grid
        h = w = int(math.isqrt(num_patch_tokens))
        if h * w != num_patch_tokens:
            # Fallback: use static grid size
            return self.backbone.patch_embed.grid_size
        return (h, w)

    def _predict(self, x):
        """Predict masks and class logits from current features."""
        q = x[:, :self.num_q, :]
        class_logits = self.class_head(q)

        # Extract patch tokens (skip queries + prefix tokens)
        x = x[:, self.num_q + self.num_prefix_tokens:, :]
        # Reshape to spatial: (B, C, H, W)
        grid_size = self._get_grid_size(x.shape[1])
        x = x.transpose(1, 2).reshape(x.shape[0], -1, *grid_size)

        # Mask logits via dot product
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw",
            self.mask_head(q),
            self.upscale(x),
        )

        return mask_logits, class_logits

    def _build_attn_mask(self, x, mask_logits):
        """Build attention mask from mask logits (query→patch only)."""
        B, N, _ = x.shape
        attn_mask = torch.ones(B, N, N, dtype=torch.bool, device=x.device)

        # Compute actual grid size from sequence length
        num_patches = N - self.num_q - self.num_prefix_tokens
        grid_size = self._get_grid_size(num_patches)
        interpolated = F.interpolate(mask_logits, grid_size, mode='bilinear')
        interpolated = interpolated.view(B, self.num_q, -1)

        # Only mask query→patch attention
        patch_start = self.num_q + self.num_prefix_tokens
        attn_mask[:, :self.num_q, patch_start:] = (interpolated > 0)

        return attn_mask

    def _disable_attn_mask(self, attn_mask, prob):
        """Randomly disable masking for some queries (annealing)."""
        if prob < 1:
            random_queries = torch.rand(
                attn_mask.shape[0], self.num_q, device=attn_mask.device
            ) > prob
            patch_start = self.num_q + self.num_prefix_tokens
            attn_mask[:, :self.num_q, patch_start:][random_queries] = True
        return attn_mask

    def _custom_attn(self, attn_module, x, mask):
        """Custom attention forward handling timm version differences."""
        B, N, C = x.shape
        qkv = attn_module.qkv(x).reshape(
            B, N, 3, attn_module.num_heads, -1
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if mask is not None:
            mask = mask[:, None, :, :]  # (B, 1, N, N) for broadcast over heads

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = attn_module.proj(x.transpose(1, 2).reshape(B, N, C))
        return x

    # ── Main forward ───────────────────────────────────────────────────

    def forward(self, x):
        # NOTE: Input normalization is handled by dataset.py
        # Alternative: uncomment below if dataset.py does NOT normalize
        # x = (x - self.pixel_mean) / self.pixel_std

        x = self.backbone.patch_embed(x)

        # Position embedding (handle timm version differences)
        if hasattr(self.backbone, '_pos_embed'):
            x, _ = self.backbone._pos_embed(x)
        elif hasattr(self.backbone, 'pos_embed'):
            x = x + self.backbone.pos_embed

        attn_mask = None
        mask_logits_list, class_logits_list = [], []

        for i, block in enumerate(self.backbone.blocks):
            # Insert queries at L1/L2 boundary
            if i == len(self.backbone.blocks) - self.num_blocks:
                x = torch.cat([
                    self.q.weight[None, :, :].expand(x.shape[0], -1, -1),
                    x,
                ], dim=1)

            # Predict intermediate masks for masked attention (training only)
            if self.masked_attn_enabled and \
               i >= len(self.backbone.blocks) - self.num_blocks:
                mask_logits, class_logits = self._predict(
                    self.backbone.norm(x)
                )
                mask_logits_list.append(mask_logits)
                class_logits_list.append(class_logits)

                attn_mask = self._build_attn_mask(x, mask_logits)
                block_idx = i - (len(self.backbone.blocks) - self.num_blocks)
                attn_mask = self._disable_attn_mask(
                    attn_mask, self.attn_mask_probs[block_idx]
                )

            # Forward through block
            # Handle timm version differences for attention
            attn = block.attn if hasattr(block, 'attn') else block.attention
            attn_out = self._custom_attn(attn, block.norm1(x), attn_mask)

            # Handle layer scaling (timm version differences)
            if hasattr(block, 'ls1'):
                x = x + block.ls1(attn_out)
            elif hasattr(block, 'layer_scale1'):
                x = x + block.layer_scale1(attn_out)
            elif hasattr(block, 'gamma_1'):
                x = x + block.gamma_1 * attn_out
            else:
                x = x + attn_out

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, 'ls2'):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, 'layer_scale2'):
                x = x + block.layer_scale2(mlp_out)
            elif hasattr(block, 'gamma_2'):
                x = x + block.gamma_2 * mlp_out
            else:
                x = x + mlp_out

        # Final prediction
        mask_logits, class_logits = self._predict(self.backbone.norm(x))
        mask_logits_list.append(mask_logits)
        class_logits_list.append(class_logits)

        return mask_logits_list, class_logits_list


if __name__ == '__main__':
    model = EoMT(num_classes=21)
    model.eval()
    x = torch.randn(1, 3, 512, 512)
    mask_logits, class_logits = model(x)
    print(f"✓ EoMT forward pass OK")
    print(f"  Mask logits: {len(mask_logits)} layers, final shape {mask_logits[-1].shape}")
    print(f"  Class logits: {len(class_logits)} layers, final shape {class_logits[-1].shape}")
