# src/models/mil_template.py

import torch
import torch.nn as nn


class MIL(nn.Module):
    """
    Base class for Multiple Instance Learning models.

    All models share one interface:
        forward(h, loss_fn=None, label=None, attn_mask=None, return_attention=False)
            -> (results_dict {'logits', 'loss'}, log_dict {'attention', 'A', ...})
    """

    def __init__(self, in_dim: int, embed_dim: int, num_classes: int):
        super().__init__()
        self.in_dim = in_dim
        self.embed_dim = embed_dim
        self.num_classes = num_classes

    def initialize_weights(self):
        """Xavier init for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def compute_loss(loss_fn: nn.Module, logits: torch.Tensor, label: torch.Tensor):
        if loss_fn is None or label is None:
            return None
        return loss_fn(logits, label)

    def forward_features(self, h: torch.Tensor, **kwargs):
        raise NotImplementedError("Subclasses must implement forward_features")

    def forward_head(self, h: torch.Tensor):
        raise NotImplementedError("Subclasses must implement forward_head")

    def forward(self, h: torch.Tensor, **kwargs):
        raise NotImplementedError("Subclasses must implement forward")
