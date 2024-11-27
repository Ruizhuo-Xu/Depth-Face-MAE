# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import random
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block

from .util.pos_embed import get_2d_sincos_pos_embed
from .arcface import ArcMarginProduct
from .models import register

@register("ViT")
class ViTForCls(nn.Module):
    """ VisionTransformer backbone for classification
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_classes=509, cls_dropout=0.9, drop_path=0.,
                 mask_ratio=None, arcface_args=None, **kwargs):
        super().__init__()
        # --------------------------------------------------------------------------
        # ViT encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, drop_path=drop_path)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # Classification specifics
        self.arcface_args = arcface_args
        if not arcface_args:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = ArcMarginProduct(**arcface_args)
        self.drop = nn.Dropout(cls_dropout)
        self.loss = nn.CrossEntropyLoss()
        # --------------------------------------------------------------------------

        self.mask_ratio = mask_ratio
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward_loss(self, pred, target):
        """
        pred: [N, C]
        target: [N,]
        """
        loss = self.loss(pred, target)
        return loss

    def forward(self, imgs, target=None, only_return_feats=False):
        if self.mask_ratio is not None and self.training:
            mask_ratio = random.uniform(0.0, self.mask_ratio)
        else:
            mask_ratio = 0.0
        latent = self.forward_encoder(imgs, mask_ratio)[:, 0]
        if only_return_feats:
            return latent
        latent = self.drop(latent)
        if not self.arcface_args:
            pred = self.head(latent)
        else:
            pred = self.head(latent, target)
        loss = self.forward_loss(pred, target)
        return loss, pred


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = {
        "img_size": 128,
        "patch_size": 16,
        "in_chans": 1,
        "embed_dim": 512,
        "depth": 12,
        "num_heads": 8,
        "mlp_ratio": 4,
    }
    model = ViTForCls(**cfg).to(device)
    from torchinfo import summary
    summary(model, (2, 1, 128, 128), device=device)