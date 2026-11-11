# src/models/sydney_mtl.py
# SydneyMTL (Kim et al., Histopathology 2026, doi:10.1111/his.70258) multi-task attention MIL과 손실
#
# 공개 코드 github.com/SMF-AI/SydneyMTL (commit f015f5c)의 sydneymtl/networks/mil.py, losses.py와 같은 구조:
#   patch features [B, N, in_dim] -> shared adaptor (Linear-GELU-Linear) [B, N, d]
#     -> 항목별 attention (Linear(d, 1) -> patch softmax) 가중합 [B, d] -> 항목별 Linear head [B, K]
#   dropout·gated attention 없음 (multitask_abmil.py와 다른 점)
# 생검 부위 입력 (논문에 없음, 우리 확장. site 0 = 부위 모름, 1~num_sites = 부위):
#   none = 논문 그대로 (부위를 쓰지 않음)
#   head = 항목별 logit에 부위·등급별 보정값을 더한다 (= head 입력에 부위 one-hot을 붙인 것과 같음). 패치 해석은 그대로
#   film = adaptor 출력에 부위별 h * (1 + gamma) + beta (attention 전) → 어떤 패치를 보고 어떻게 읽을지가 부위에 따라 바뀜
#   두 방식 모두 보정값을 0으로 시작하고 site 0은 보정 0으로 고정 → 부위 모름 = 부위 없는 모델과 같은 계산
# 손실 (항목마다 하나):
#   la  = logit-adjusted CE (Menon et al. 2021): CE(f(x) + tau * log prior). 예측은 보정 전 f(x)로 한다
#   ce  = 일반 CE
#   wce = 등급 빈도 역수 가중 CE. 가중치 = 1/count를 평균 1로 정규화 (공개 코드 compute_class_weight와 같음).
#         batch 1에서도 효과가 있도록 표본별로 w[y] * CE를 곱한다 (nn.CrossEntropyLoss(weight)는 batch 1이면 가중치가 약분됨)
# train fold에 없는 등급은 예측에서 제외한다 (la에서는 학습 신호가 없어 보정 전 logit이 임의의 값으로 남기 때문).

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ("la", "ce", "wce")
SITE_MODES = ("none", "head", "film")


def _zero_embedding(n: int, dim: int) -> nn.Embedding:
    """Embedding that starts at 0 everywhere; row 0 (site unknown) stays 0 (padding_idx = no gradient)."""
    emb = nn.Embedding(n, dim, padding_idx=0)
    nn.init.zeros_(emb.weight)
    return emb


class SydneyMTL(nn.Module):

    def __init__(self, in_dim: int, task_classes: Dict[str, int], adaptor_dim: int = 256,
                 site_mode: str = "none", num_sites: int = 0):
        super().__init__()
        if site_mode not in SITE_MODES:
            raise ValueError(f"site_mode must be one of {SITE_MODES}, got '{site_mode}'")
        self.tasks = list(task_classes)
        self.site_mode = site_mode
        self.adaptor = nn.Sequential(
            nn.Linear(in_dim, adaptor_dim),
            nn.GELU(),
            nn.Linear(adaptor_dim, adaptor_dim),
        )
        self.attention = nn.ModuleDict({t: nn.Linear(adaptor_dim, 1) for t in self.tasks})
        self.classifiers = nn.ModuleDict({t: nn.Linear(adaptor_dim, k) for t, k in task_classes.items()})
        # 부위 모듈은 기존 층 뒤에 만들어서 none일 때와 같은 seed면 기존 층 초기값이 같다
        if site_mode == "head":
            self.site_bias = nn.ModuleDict({t: _zero_embedding(num_sites + 1, k) for t, k in task_classes.items()})
        elif site_mode == "film":
            self.site_film = _zero_embedding(num_sites + 1, 2 * adaptor_dim)

    def forward(self, x: torch.Tensor, site: Optional[torch.Tensor] = None
                ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """x: [B, N, in_dim], site: [B] long (0 = unknown, None = all unknown)
        -> logits {task: [B, K]}, attention {task: [B, N] softmax over patches}"""
        if site is None:
            site = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        h = self.adaptor(x)
        if self.site_mode == "film":
            gamma, beta = self.site_film(site).unsqueeze(1).chunk(2, dim=-1)  # [B, 1, d] 각각
            h = h * (1 + gamma) + beta
        logits, attention = {}, {}
        for t in self.tasks:
            a = torch.softmax(self.attention[t](h).squeeze(-1), dim=-1)
            logits[t] = self.classifiers[t](torch.einsum("bn,bnd->bd", a, h))
            if self.site_mode == "head":
                logits[t] = logits[t] + self.site_bias[t](site)
            attention[t] = a
        return logits, attention


class GradeLoss(nn.Module):
    """One task's loss built from that task's train-fold labels."""

    def __init__(self, kind: str, train_labels: torch.Tensor, num_classes: int, tau: float = 0.75):
        super().__init__()
        if kind not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got '{kind}'")
        self.kind, self.tau = kind, tau
        counts = torch.bincount(train_labels.long(), minlength=num_classes).float()
        prior = counts / counts.sum()
        # 0건 등급은 log(0) 대신 아주 작은 prior로 둔다 (val에 그 등급이 있어도 loss가 inf가 되지 않도록)
        self.register_buffer("log_prior", torch.log(prior.clamp_min(1e-8)))
        weights = 1.0 / (counts + 1e-6)
        self.register_buffer("weights", weights / weights.mean())
        self.register_buffer("unseen", counts == 0)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "la":
            return F.cross_entropy(logits + self.tau * self.log_prior, target)
        if self.kind == "wce":
            return (self.weights[target] * F.cross_entropy(logits, target, reduction="none")).mean()
        return F.cross_entropy(logits, target)

    def predict_proba(self, logits: torch.Tensor) -> torch.Tensor:
        """Grade probabilities from raw logits, with grades unseen in training masked out."""
        return torch.softmax(logits.masked_fill(self.unseen, float("-inf")), dim=-1)
