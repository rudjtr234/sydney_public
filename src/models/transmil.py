# src/models/transmil.py
# Thyroid_Mutation_model_v2/src/models/transmil/model.py 와 동일

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import PositionalEncoding
from .mil_template import MIL


class TransMILModel(MIL):
    """
    Transformer-based MIL model inspired by TransMIL.

    For stability with very large bags, long inputs are truncated to `max_tokens`
    before self-attention (random subset in training, evenly spaced subset in eval).
    """

    def __init__(self,
                 in_dim: int = 1536,
                 embed_dim: int = 512,
                 n_heads: int = 8,
                 n_layers: int = 2,
                 ffn_dim: int = 1024,
                 dropout: float = 0.1,
                 max_tokens: int = 2048,
                 num_classes: int = 2,
                 **kwargs):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.max_tokens = max_tokens

        self.patch_embed = nn.Linear(in_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=max_tokens + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)

        self.initialize_weights()
        nn.init.normal_(self.cls_token, std=0.02)

    def _truncate_instances(self, h: torch.Tensor):
        batch_size, n_instances, _ = h.shape
        if self.max_tokens is None or n_instances <= self.max_tokens:
            indices = torch.arange(n_instances, device=h.device).unsqueeze(0).repeat(batch_size, 1)
            return h, indices

        if self.training:
            sampled_idx = torch.randperm(n_instances, device=h.device)[: self.max_tokens]
            sampled_idx, _ = torch.sort(sampled_idx)
            indices = sampled_idx.unsqueeze(0).repeat(batch_size, 1)
        else:
            lin_idx = torch.linspace(0, n_instances - 1, steps=self.max_tokens, device=h.device)
            indices = lin_idx.long().unsqueeze(0).repeat(batch_size, 1)

        gather_idx = indices.unsqueeze(-1).expand(-1, -1, h.size(-1))
        h = torch.gather(h, dim=1, index=gather_idx)
        return h, indices

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        h = self.patch_embed(h)
        h, kept_indices = self._truncate_instances(h)

        batch_size = h.size(0)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_token, h], dim=1)   # [B, N+1, D]

        tokens = tokens.transpose(0, 1)             # [N+1, B, D]
        tokens = self.pos_encoder(tokens)
        tokens = tokens.transpose(0, 1)             # [B, N+1, D]

        encoded = self.transformer(tokens)
        cls_repr = encoded[:, 0]
        patch_repr = encoded[:, 1:]

        attn_logits = torch.einsum("bnd,bd->bn", patch_repr, cls_repr)
        attn_logits = attn_logits / math.sqrt(cls_repr.size(-1))
        A = F.softmax(attn_logits, dim=-1).unsqueeze(1)

        log_dict = {
            "attention": attn_logits.unsqueeze(1) if return_attention else None,
            "A": A if return_attention else None,
            "kept_indices": kept_indices,
        }
        return cls_repr, log_dict

    def forward_head(self, h: torch.Tensor):
        return self.classifier(h)

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False):
        wsi_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(wsi_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        results_dict = {"logits": logits, "loss": cls_loss}
        log_dict["loss"] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict["slide_feats"] = wsi_feats
        return results_dict, log_dict
