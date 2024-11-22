# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from .util.pos_embed import get_2d_sincos_pos_embed
from .models import register

@register("ViT")
class ViTForCls(nn.Module):
    """ VisionTransformer backbone for classification
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 num_classes=509, cls_dropout=0.9, **kwargs):
        super().__init__()
        # --------------------------------------------------------------------------
        # ViT encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------
        
        # --------------------------------------------------------------------------
        # Classification specifics
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.drop = nn.Dropout(cls_dropout)
        self.loss = nn.CrossEntropyLoss()
        # --------------------------------------------------------------------------

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

    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

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
        latent = self.forward_encoder(imgs)[:, 0]
        if only_return_feats:
            return latent
        latent = self.drop(latent)
        pred = self.head(latent)
        loss = self.forward_loss(pred, target)
        return loss, latent


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