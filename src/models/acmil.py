# src/models/acmil.py
# Thyroid_Mutation_model_v2/src/models/acmil/model.py 와 동일

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import create_mlp
from .mil_template import MIL


class ACMILModel(MIL):
    """
    ACMIL (Attention-Challenging MIL, ECCV 2024: https://github.com/dazhangyu123/ACMIL).

    - Multiple attention branches to diversify attention
    - Stochastic top-k masking during training to force learning from hard instances
    - Bag representation = mean of multi-branch attention-pooled features
    """

    def __init__(self,
                 in_dim: int = 1536,
                 embed_dim: int = 512,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 num_branches: int = 4,
                 topk_ratio: float = 0.1,
                 num_classes: int = 2,
                 **kwargs):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.num_branches = num_branches
        self.topk_ratio = topk_ratio

        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        attn_dim = embed_dim // 2
        self.attn_branches = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, attn_dim), nn.Tanh(), nn.Linear(attn_dim, 1))
            for _ in range(num_branches)
        ])
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def _compute_branch_attention(self, h: torch.Tensor, branch: nn.Module, mask: torch.Tensor = None):
        A = branch(h).transpose(-2, -1)  # [B, 1, N]
        if mask is not None:
            A = A.masked_fill(mask.unsqueeze(1), torch.finfo(A.dtype).min)
        return A

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only: bool = True):
        h_embed = self.patch_embed(h)
        # Use first branch for single-branch attention output (eval / visualization)
        A = self._compute_branch_attention(h_embed, self.attn_branches[0], mask=None)
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(1) * torch.finfo(A.dtype).min
        if attn_only:
            return A
        return h_embed, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        h_embed = self.patch_embed(h)  # [B, N, D]
        B, N, D = h_embed.shape

        topk = max(1, int(N * self.topk_ratio))
        branch_feats = []
        all_A = []

        for i, branch in enumerate(self.attn_branches):
            # Stochastic top-k masking: mask the top-k attended patches of previous branch
            mask = None
            if self.training and i > 0 and len(all_A) > 0:
                prev_A = F.softmax(all_A[-1], dim=-1).squeeze(1)  # [B, N]
                topk_idx = prev_A.topk(topk, dim=-1).indices
                mask = torch.zeros(B, N, dtype=torch.bool, device=h.device)
                mask.scatter_(1, topk_idx, True)

            A_raw = self._compute_branch_attention(h_embed, branch, mask=mask)
            if attn_mask is not None:
                A_raw = A_raw + (1 - attn_mask).unsqueeze(1) * torch.finfo(A_raw.dtype).min

            A = F.softmax(A_raw, dim=-1)
            branch_feats.append(torch.bmm(A, h_embed).squeeze(1))
            all_A.append(A_raw)

        bag_feats = torch.stack(branch_feats, dim=1).mean(dim=1)  # [B, D]
        log_dict = {
            "attention": all_A[0] if return_attention else None,
            "A": F.softmax(all_A[0], dim=-1) if return_attention else None,
        }
        return bag_feats, log_dict

    def forward_head(self, h: torch.Tensor) -> torch.Tensor:
        return self.classifier(h)

    def forward(self, h: torch.Tensor,
                loss_fn: nn.Module = None,
                label: torch.LongTensor = None,
                attn_mask=None,
                return_attention: bool = False,
                return_slide_feats: bool = False):
        bag_feats, log_dict = self.forward_features(h, attn_mask=attn_mask, return_attention=return_attention)
        logits = self.forward_head(bag_feats)
        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        results_dict = {"logits": logits, "loss": cls_loss}
        log_dict["loss"] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict["slide_feats"] = bag_feats
        return results_dict, log_dict
