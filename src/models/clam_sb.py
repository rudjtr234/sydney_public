# src/models/clam_sb.py
# Thyroid_Mutation_model_v2/src/models/clam_sb/model.py 와 동일

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GlobalAttention, GlobalGatedAttention, create_mlp
from .mil_template import MIL


class CLAMSBModel(MIL):
    """
    CLAM-SB style MIL model.

    Note:
        This implementation focuses on the bag-level branch to keep compatibility
        with the training loop. It adds an auxiliary max-instance branch
        to emulate CLAM's instance-aware behavior while keeping a single loss term
        (no instance clustering loss).
    """

    def __init__(self,
                 in_dim: int = 1536,
                 embed_dim: int = 512,
                 attn_dim: int = 384,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 gate: bool = True,
                 num_classes: int = 2,
                 **kwargs):
        super().__init__(in_dim=in_dim, embed_dim=embed_dim, num_classes=num_classes)
        self.patch_embed = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        attn_layer = GlobalGatedAttention if gate else GlobalAttention
        self.global_attn = attn_layer(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.instance_classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def forward_attention(self, h: torch.Tensor, attn_mask=None, attn_only: bool = True):
        h = self.patch_embed(h)
        A = self.global_attn(h)          # [B, N, 1]
        A = torch.transpose(A, -2, -1)   # [B, 1, N]
        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        if attn_only:
            return A
        return h, A

    def forward_features(self, h: torch.Tensor, attn_mask=None, return_attention: bool = True):
        h, A_raw = self.forward_attention(h, attn_mask=attn_mask, attn_only=False)
        A = F.softmax(A_raw, dim=-1)
        bag_feats = torch.bmm(A, h).squeeze(dim=1)
        instance_logits = self.instance_classifier(h)
        log_dict = {
            "attention": A_raw if return_attention else None,
            "A": A if return_attention else None,
            "instance_logits": instance_logits,
            "max_instance_logits": instance_logits.max(dim=1).values,
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
        bag_logits = self.forward_head(bag_feats)
        logits = 0.5 * (bag_logits + log_dict["max_instance_logits"])
        cls_loss = MIL.compute_loss(loss_fn, logits, label)

        results_dict = {"logits": logits, "loss": cls_loss}
        log_dict["loss"] = cls_loss.item() if cls_loss is not None else -1
        if return_slide_feats:
            log_dict["slide_feats"] = bag_feats
        return results_dict, log_dict
