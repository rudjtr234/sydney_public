# src/models/multitask_abmil.py
# Sydney 항목별 Grade 예측용 Multi-task ABMIL
#
#   patch features [B, N, in_dim] -> shared adapter [B, N, E]  (모든 항목이 공유, 5개 항목 gradient를 다 받는다)
#     -> 항목별 gated attention pooling [B, E] -> 항목별 head
#
# head "ce":    Grade logit K개, softmax + cross entropy
# head "coral": 공유 weight 1개 + 순서 bias K-1개 -> P(grade > k) logit, CORAL loss
#               (Cao et al. 2020, https://arxiv.org/abs/1901.07884)
# Sydney CODE는 항목별 예측 Grade를 이어 붙인 결과일 뿐이고, CODE 자체를 클래스로 학습하지 않는다.

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import GlobalAttention, GlobalGatedAttention, create_mlp

HEADS = ("ce", "coral")

MULTITASK_ABMIL_DEFAULTS: Dict[str, Any] = {
    "embed_dim": 256,
    "attn_dim": 128,
    "num_fc_layers": 1,
    "dropout": 0.25,
    "gate": True,
}


class CoralHead(nn.Module):
    """One weight vector shared by all thresholds + K-1 biases -> logits of P(grade > k), k = 0..K-2."""

    def __init__(self, in_dim: int, num_grades: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        # coral-pytorch CoralLayer와 같은 초기값 (내림차순)
        self.bias = nn.Parameter(torch.arange(num_grades - 1, 0, -1).float() / (num_grades - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) + self.bias


class MultiTaskABMIL(nn.Module):

    def __init__(self,
                 in_dim: int,
                 tasks: List[str],
                 num_grades: int = 4,
                 head: str = "ce",
                 embed_dim: int = 256,
                 attn_dim: int = 128,
                 num_fc_layers: int = 1,
                 dropout: float = 0.25,
                 gate: bool = True):
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"head must be one of {HEADS}, got '{head}'")
        self.tasks = list(tasks)
        self.num_grades = num_grades
        self.head = head

        self.adapter = create_mlp(
            in_dim=in_dim,
            hid_dims=[embed_dim] * (num_fc_layers - 1),
            dropout=dropout,
            out_dim=embed_dim,
            end_with_fc=False,
        )
        attn_func = GlobalGatedAttention if gate else GlobalAttention
        self.attn = nn.ModuleDict({t: attn_func(L=embed_dim, D=attn_dim, dropout=dropout, num_classes=1)
                                   for t in self.tasks})
        self.heads = nn.ModuleDict({t: nn.Linear(embed_dim, num_grades) if head == "ce"
                                    else CoralHead(embed_dim, num_grades) for t in self.tasks})
        self.initialize_weights()

    def initialize_weights(self):
        """Xavier init for linear layers (CORAL biases keep their ordered init)."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, h: torch.Tensor, return_attention: bool = False
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        h: [B, N, in_dim]
        -> logits {task: [B, K] (ce) or [B, K-1] (coral)}, attention {task: [B, 1, N] softmax} (if requested)
        """
        h = self.adapter(h)
        logits, attention = {}, {}
        for t in self.tasks:
            A = F.softmax(self.attn[t](h).transpose(-2, -1), dim=-1)  # [B, 1, N]
            logits[t] = self.heads[t](torch.bmm(A, h).squeeze(dim=1))
            if return_attention:
                attention[t] = A
        return logits, attention


def coral_levels(y: torch.Tensor, num_grades: int) -> torch.Tensor:
    """Grade y -> binary targets [y > 0, y > 1, ..., y > K-2]."""
    return (y.unsqueeze(1) > torch.arange(num_grades - 1, device=y.device)).float()


def grade_probs(logits: torch.Tensor, head: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """logits of one task -> (grade probabilities [B, K], predicted grade [B])."""
    if head == "ce":
        probs = torch.softmax(logits, dim=1)
        return probs, probs.argmax(dim=1)
    p_gt = torch.sigmoid(logits)                                      # P(grade > k)
    ones, zeros = torch.ones_like(p_gt[:, :1]), torch.zeros_like(p_gt[:, :1])
    cum = torch.cat([ones, p_gt, zeros], dim=1)
    probs = (cum[:, :-1] - cum[:, 1:]).clamp_min(0)                   # P(grade = k)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return probs, (p_gt > 0.5).sum(dim=1)                             # CORAL 예측 = 넘은 임계값 개수


class MultiTaskLoss(nn.Module):
    """
    L_total = sum_t w_t * L_t / sum_t w_t over the tasks.
    Missing grades (label < 0) are left out of their task's loss only; a task with no valid label adds 0.
    class_weights [T, K]: per-sample weight w[t, y] (applied per sample, so it also works with batch size 1).
    """

    def __init__(self, tasks: List[str], head: str, num_grades: int,
                 class_weights: Optional[torch.Tensor] = None,
                 task_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.tasks = list(tasks)
        self.head = head
        self.num_grades = num_grades
        task_weights = task_weights or {}
        self.register_buffer("task_weights", torch.tensor([float(task_weights.get(t, 1.0)) for t in self.tasks]))
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: Dict[str, torch.Tensor], labels: torch.Tensor
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """labels: [B, T] grades, -1 = missing. Returns (total loss, {task: loss} for tasks with a valid label)."""
        total = sum(z.sum() * 0 for z in logits.values())  # 모든 항목이 결측이어도 backward가 되도록
        per_task = {}
        for i, t in enumerate(self.tasks):
            y = labels[:, i]
            valid = y >= 0
            if not valid.any():
                continue
            z, y = logits[t][valid], y[valid]
            if self.head == "ce":
                loss = F.cross_entropy(z, y, reduction="none")
            else:
                loss = F.binary_cross_entropy_with_logits(z, coral_levels(y, self.num_grades), reduction="none").sum(1)
            if self.class_weights is not None:
                loss = loss * self.class_weights[i, y]
            loss = loss.mean()
            per_task[t] = loss.item()
            total = total + self.task_weights[i] * loss
        return total / self.task_weights.sum(), per_task


def create_multitask_abmil(in_dim: int, tasks: List[str], num_grades: int = 4, head: str = "ce",
                           **overrides) -> Tuple[MultiTaskABMIL, Dict[str, Any]]:
    """Returns (model, resolved_kwargs). Unknown override keys raise (same rule as create_mil_model)."""
    kwargs = deepcopy(MULTITASK_ABMIL_DEFAULTS)
    unknown = sorted(set(overrides) - set(kwargs))
    if unknown:
        raise ValueError(f"Unknown kwargs for multitask_abmil: {unknown}. Allowed: {sorted(kwargs)}")
    kwargs.update({k: v for k, v in overrides.items() if v is not None})

    model_kwargs = {"in_dim": in_dim, "tasks": list(tasks), "num_grades": num_grades, "head": head, **kwargs}
    return MultiTaskABMIL(**model_kwargs), model_kwargs
