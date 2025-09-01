


# ========= helpers =========
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from collections import deque

IGNORE_INDEX = -100  # HF 기본

import torch

from sklearn.linear_model import LinearRegression
import numpy as np

from sklearn.linear_model import LinearRegression
import numpy as np


# pip install torch
from dataclasses import dataclass
import math, torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters



import math, random
from typing import List, Tuple, Optional, Dict, Iterator
import torch
from torch.utils.data import Sampler



from collections import deque
import torch
from typing import Optional, Tuple

class RollingQuantile:
    """최근 window_size개만 유지하며 q-분위수 계산."""
    def __init__(self, q: float, window_size: int = 128):
        assert 0.0 < q < 1.0
        self.q = float(q)
        self.window_size = int(window_size)
        self.win = deque(maxlen=self.window_size)

    def update(self, x: torch.Tensor):
        """x: arbitrary shape tensor -> flatten해서 최근 값으로 덧붙임"""
        x = x.detach().to("cpu").view(-1)
        if x.numel() == 0:
            return
        x = x[torch.isfinite(x)]
        for v in x.tolist():
            self.win.append(v)

    def value(self) -> Optional[float]:
        if len(self.win) == 0:
            return None
        buf = torch.tensor(list(self.win), dtype=torch.float32)
        return float(torch.quantile(buf, self.q).item())

    def as_tensor(self) -> Optional[torch.Tensor]:
        if len(self.win) == 0:
            return None
        return torch.tensor(list(self.win), dtype=torch.float32)

    def reset(self):
        self.win.clear()

    def set_window_size(self, window_size: int):
        data = list(self.win)
        self.window_size = int(window_size)
        self.win = deque(data[-self.window_size:], maxlen=self.window_size)

def _normalize(x: torch.Tensor, how: Optional[str]) -> torch.Tensor:
    if how is None:
        return x.float()
    x = x.float()
    if how == "mad":
        med = torch.median(x)
        mad = torch.median(torch.abs(x - med)) + 1e-12
        return (x - med) / (mad * 1.4826)
    # z-score
    mu = x.mean()
    sd = x.std(unbiased=False) + 1e-12
    return (x - mu) / sd


class QuantileState:
    """
    - EMA 난이도 점수(ema)
    - prior 혼합
    - RollingQuantile 기반 t33/t66 (최근 window_size개만으로 계산, 매 배치 갱신)
    """
    def __init__(self, n_items:int, beta:float=0.0, init_score:float=1.0,
                 device:str="cpu",
                 priors: Optional[torch.Tensor] = None,
                 prior_norm: Optional[str] = "zscore",
                 prior_mix: float = 0.0, # 0.5, 
                 init_from_prior: bool = True,
                 # 슬라이딩 윈도우/스무딩 설정
                 window_size:int = 256,
                 th_ema_alpha: float = 0.2,  # 0이면 스무딩 없음, >0이면 경계 EMA
                 prior_t33: Optional[float] = None,
                 prior_t66: Optional[float] = None):
        self.n = int(n_items)
        self.device = device
        self.beta = float(beta)

        # prior & ema 초기화
        if init_from_prior and (priors is not None):
            p = priors.detach().to(device=device, dtype=torch.float32)
            # if prior_norm is not None:
            #     p = _normalize(p, prior_norm)
            self.prior = p
            self.ema = self.prior.clone()
        else:
            self.prior = torch.zeros(n_items, device=device, dtype=torch.float32)
            self.ema = torch.full((n_items,), float(init_score), device=device)
        self.seen = torch.zeros(n_items, dtype=torch.bool, device=device)
        self.prior_mix = float(prior_mix)

        # 롤링 분위수 추정기
        self.rq_low  = RollingQuantile(q=0.33, window_size=window_size)
        self.rq_high = RollingQuantile(q=0.66, window_size=window_size)

        # 경계 캐시(옵션: EMA 스무딩)
        self._t33: Optional[float] = None
        self._t66: Optional[float] = None
        self._th_ema_alpha = float(th_ema_alpha)

        # 초기 경계 (prior가 있으면 그걸로 세팅)
        if prior_t33 is not None and prior_t66 is not None:
            self._t33, self._t66 = float(prior_t33), float(prior_t66)
        elif priors is not None:
            ps = self.prior.detach().to("cpu")
            t33, t66 = torch.quantile(ps, torch.tensor([0.33, 0.66])).tolist()
            self._t33, self._t66 = float(t33), float(t66)


    def set_prior_mix(self, prior_mix: float):
        self.prior_mix = float(max(0.0, min(1.0, prior_mix)))

    def schedule_prior_mix(self, t: float, lam0: float = 0.6, lam1: float = 0.2, mode: str = "cos"):
        t = max(0.0, min(1.0, t))
        if mode == "cos":
            a = 0.5 - 0.5*torch.cos(torch.tensor(t)*torch.pi)
            lam = (1-a)*lam0 + a*lam1
        else:
            lam = (1-t)*lam0 + t*lam1
        self.set_prior_mix(float(lam))

    @torch.no_grad()
    def update_losses(self, idxs: torch.Tensor, per_sample_losses: torch.Tensor):
        """배치마다: 오직 스코어(EMA)와 seen만 갱신. 경계 갱신은 절대 하지 않음."""
        old = self.ema[idxs]
        self.ema[idxs] = self.beta * old + (1.0 - self.beta) * per_sample_losses
        self.seen[idxs] = True
    @torch.no_grad()
    def thresholds(self) -> tuple[float, float]:
        return float(self._t33), float(self._t66)

    @torch.no_grad()
    def score_for(self, inds: torch.Tensor) -> torch.Tensor:
        """버킷팅/ε 스코어: s = (1-λ)*ema + λ*prior"""
        if self.prior_mix <= 1e-12:            
            return self.ema[inds]
        if self.prior_mix >= 1.0 - 1e-12:
            return self.prior[inds]
        return (1 - self.prior_mix) * self.ema[inds] + self.prior_mix * self.prior[inds]
    
    
    @torch.no_grad()
    def set_thresholds(self, t33: float, t66: float):
        """샘플러가 풀 기반으로 계산한 경계를 캐시에 저장(최소 간격 보호 + 선택적 스무딩)."""
        if t66 - t33 < 1e-8:
            mid = 0.5*(t33+t66)
            t33, t66 = mid-5e-9, mid+5e-9

        if (self._t33 is None) or (self._t66 is None) or (self._th_ema_alpha <= 0.0):
            self._t33, self._t66 = float(t33), float(t66)
        else:
            a = self._th_ema_alpha
            self._t33 = (1-a)*self._t33 + a*float(t33)
            self._t66 = (1-a)*self._t66 + a*float(t66)

    @torch.no_grad()
    def refresh_thresholds_from_pool(self, pool_indices: list[int] | torch.Tensor):
        """<<여기가 핵심>> 남은 풀의 점수 분포로부터 (t33,t66) 계산 후 set_thresholds()."""
        if isinstance(pool_indices, list):
            if len(pool_indices) == 0:
                return
            inds = torch.tensor(pool_indices, device=self.device, dtype=torch.long)
        else:
            if pool_indices.numel() == 0:
                return
            inds = pool_indices.to(self.device, dtype=torch.long)

        s = self.score_for(inds)  # (prior mix 반영된) 최종 난이도 점수
        t33, t66 = torch.quantile(
            s, torch.tensor([0.33, 0.66], device=s.device)
        ).tolist()
        self.set_thresholds(t33, t66)
    @torch.no_grad()
    def classify_indices(self, inds: torch.Tensor) -> list[str]:
        """
        inds: [B] int64 on device
        return: ["L"|"M"|"H"] * B
        """
        s = self.score_for(inds)             # EMA(+prior mix) 점수
        t33, t66 = self.thresholds()         # K-step 캐시된 경계 (pool 기반으로 갱신됨)
        lbl = []
        s_list = s.tolist()
        for v in s_list:
            if v < t33:      lbl.append("L")
            elif v >= t66:   lbl.append("H")
            else:            lbl.append("M")
        return lbl
        
def power_mean(x: torch.Tensor, p: float) -> torch.Tensor:
    if p == 0.0:  # geometric mean
        return torch.exp(torch.mean(torch.log(x.clamp_min(1e-8))))
    return (x.clamp_min(1e-8).pow(p).mean()).pow(1.0 / p)

class EpsilonPolicy:
    """
    - 연속 ε 매핑 (sigmoid with temperature τ) : eps_min ~ eps_max 사이에서 mid 중심으로 sharpen
    - 배치 집계는 산술평균 대신 p-mean (p>=2 권장)으로 동적 폭 확대
    - 선택: 배치 이질성(표준편차) 가산항 k*std (k=0이면 비활성)
    """
    def __init__(
        self,
        rho_base: float = 0.05,
        eps_min: float = 0.6,
        eps_mid: float = 1.0,
        eps_max: float = 1.6,
        tau: float = 0.25,      # 작을수록 경계 샤프
        p_mean: float = 2.0,    # 2~4 권장
        k_std: float = 0.0,      # 0이면 분산 가산 비활성, 0.3~0.6 시도 가능
        mode: str="conti", # bucket
    ):
        self.rho_base = float(rho_base)
        self.eps_L = float(eps_min)
        self.eps_M = float(eps_mid)
        self.eps_H = float(eps_max)
        self.tau = float(tau)
        self.p_mean = float(p_mean)
        self.k_std = float(k_std)
        self.mode = mode
        # 연속 모드용 하이퍼

        # (옵션) 진행도에 따라 eps_max를 키우고 싶으면 set_progress 사용
        self._progress = 0.0

    def set_progress(self, t: float):
        """0~1 진행도(토큰/스텝 기반) 연결해서 eps_max를 점진적으로 키우고 싶을 때 사용."""
        self._progress = max(0.0, min(1.0, float(t)))
        # 예: 초반 안전장치 (eps_max 1.4 → 1.6로 상승)
        base, top = 1.4, self.eps_max
        w = 0.5 * (1 - math.cos(math.pi * self._progress))
        self._eps_max_eff = (1 - w) * base + w * top
    @property
    def eps_max_eff(self):
        return getattr(self, "_eps_max_eff", self.eps_max)

    @torch.no_grad()
    def rho_eff(self, state, batch_indices: list[int], return_eps: bool=False):
        if len(batch_indices) == 0:
            return (self.rho_base, None) if return_eps else self.rho_base

        device = state.device
        inds = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
        s = state.score_for(inds)  # EMA(+prior mix)

        if self.mode == "bucket":
            # --- 경계 기반 L/M/H ---
            t33, t66 = state.thresholds()      # K-step 갱신된 캐시 사용 (계단형)
            L = s < t33
            H = s >= t66
            M = ~(L | H)
            eps = torch.empty_like(s, dtype=torch.float32)
            eps[L] = self.eps_L
            eps[M] = self.eps_M
            eps[H] = self.eps_H
            eps_agg = power_mean(eps, self.p_mean)

        else:
            # --- 연속(중앙값 기준) ---
            m = torch.median(s)
            z = (s - m) / (self.tau + 1e-8)
            w = torch.sigmoid(z)  # [0,1]
            eps_hi = self.eps_M + w * (self.eps_H - self.eps_M)   # m↑일수록 H쪽
            eps_lo = self.eps_M - (1 - w) * (self.eps_M - self.eps_L)
            eps = torch.where(s >= m, eps_hi, eps_lo)
            eps_agg = power_mean(eps, self.p_mean)
            if self.k_std > 0:
                eps_agg = eps_agg + self.k_std * eps.std(unbiased=False)

        rho_eff = float((eps_agg * self.rho_base).item())
        return (rho_eff, eps) if return_eps else rho_eff
    
    
    # (선택) 디버깅/로깅용
    @torch.no_grad()
    def rho_eff_with_stats(self, state, batch_indices: list[int]):
        device = state.device
        inds = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
        s = state.score_for(inds)
        
        # 1) static
        # t33, t66 = state.thresholds(inds)
        # L = s < t33
        # H = s >= t66
        # M = ~(L | H)
        # scale = torch.zeros_like(s, dtype=torch.float32)
        # scale[L] = self.eps_min
        # scale[M] = self.eps_mid
        # scale[H] = self.eps_max_eff
        # print('scale',"score", scale, s)
        # 2) gradually
        m = torch.median(s)
        z = (s - m) / (self.tau + 1e-8)
        w = torch.sigmoid(z)
        eps_hi = self.eps_mid + w * (self.eps_max_eff - self.eps_mid)
        eps_lo = self.eps_mid - (1 - w) * (self.eps_mid - self.eps_min)
        eps = torch.where(s >= m, eps_hi, eps_lo)
        eps_agg = power_mean(eps, self.p_mean)
        if self.k_std > 0:
            eps_agg = eps_agg + self.k_std * eps.std(unbiased=False)
        rho = float((eps_agg * self.rho_base).item())
        stats = {
            "eps_mean": float(eps.mean().item()),
            "eps_std": float(eps.std(unbiased=False).item()),
            "score_med": float(m.item()),
            "p_gt_med": float((s >= m).float().mean().item()),
            "eps_agg": float(eps_agg.item()),
        }
        return rho, stats
        

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

# ---- LoRA 이름 패턴(원하면 켜고, 기본은 requires_grad=True 전체 사용) ----
LORA_NAME_KEYS = [
    ".lora_A.", ".lora_B.",
    ".lora_embedding_A.", ".lora_embedding_B.",
    ".lora_magnitude_vector."
]

def collect_trainable_params(
    model: nn.Module,
    only_lora: bool = False,
) -> List[nn.Parameter]:
    params = []
    if only_lora:
        for name, p in model.named_parameters():
            if p.requires_grad and any(k in name for k in LORA_NAME_KEYS):
                params.append(p)
        if not params:
            # 로라 패턴이 없으면 fallback: requires_grad=True 전체
            params = [p for _, p in model.named_parameters() if p.requires_grad]
    else:
        params = [p for _, p in model.named_parameters() if p.requires_grad]
    return params

def causal_lm_loss(model: nn.Module, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    """HF causal LM 호환: labels 없으면 input_ids로 채움."""
    batch = dict(inputs)
    if "labels" not in batch or batch["labels"] is None:
        batch["labels"] = batch["input_ids"].clone()
    outputs = model(**batch)
    if outputs.loss is None:
        raise RuntimeError("loss is None (labels 필요).")
    return outputs.loss

def _zeros_like_params(params: List[torch.Tensor]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]

def _backup_params(params: List[nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]

@torch.no_grad()
def _apply_delta(params: List[nn.Parameter], delta: List[torch.Tensor], coef: float) -> None:
    for p, d in zip(params, delta):
        p.add_(coef * d)

def _elementwise_sqrt(t: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(torch.clamp(t, min=0.0) + eps)

def _maybe_init_diag_H(
    params: List[nn.Parameter],
    state: Dict[str, List[torch.Tensor]],
    init_value: float = 1.0,
) -> None:
    if "diag_H" not in state or state["diag_H"] is None:
        state["diag_H"] = [torch.full_like(p, init_value) for p in params]

def _sample_u_like(params: List[nn.Parameter]) -> List[torch.Tensor]:
    return [torch.randn_like(p) for p in params]

def _mul_lists(a: List[torch.Tensor], b: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x * y for x, y in zip(a, b)]

def _sqr_list(a: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x * x for x in a]

def _scale_list(a: List[torch.Tensor], s: float) -> List[torch.Tensor]:
    return [x * s for x in a]

def _abs_list(a: List[torch.Tensor]) -> List[torch.Tensor]:
    return [x.abs() for x in a]

def _ema_update(old: List[torch.Tensor], new: List[torch.Tensor], alpha: float) -> List[torch.Tensor]:
    return [(1 - alpha) * o + alpha * n for o, n in zip(old, new)]

@torch.no_grad()
def hizoo_spsa_diag_step(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    state: Dict[str, List[torch.Tensor]],
    *,
    mu: float = 1e-3,          # perturb scale μ
    alpha: float = 1e-6,       # EMA rate for diag(H)
    n: int = 1,                # n-SPSA samples
    eps: float = 1e-8,
    only_lora: bool = True,
    base_loss: Optional[torch.Tensor] = None,   # 이미 계산된 L(θ)를 줄 수 있음(없으면 내부에서 1회 계산)
) -> Tuple[List[torch.Tensor], Dict[str, List[torch.Tensor]], Dict[str, float]]:
    """
    HiZOO 1 step:
      - 입력: model, inputs, (state에 diag_H 보관)
      - 출력: (g_spsa_est, state, metrics)
        * g_spsa_est: n-SPSA 평균(LoRA 등 학습 파라미터 shape 그대로)
        * state["diag_H"]: EMA 업데이트된 대각 Hessian 추정
        * metrics: 손실/델타 등 로그용
    """
    # 1) 학습 파라미터 추출
    params = collect_trainable_params(model, only_lora=only_lora)
    if len(params) == 0:
        raise RuntimeError("No trainable params (requires_grad=True).")

    _maybe_init_diag_H(params, state, init_value=1.0)
    diag_H = state["diag_H"]  # List[tensor], ≥0로 유지 권장

    # 2) 현재 loss L(θ) (재사용 가능)
    was_training = model.training
    model.eval()  # 드롭아웃/정규화 고정
    if base_loss is None:
        base_loss = causal_lm_loss(model, inputs)
    base_loss_val = float(base_loss.detach().cpu())

    # 3) 준비: Σ^{1/2}, Σ^{-1/2}, Σ^{-1}
    #    Σ = H^{-1} (근사) → Σ^{1/2} = 1/sqrt(H),  Σ^{-1/2} = sqrt(H),  Σ^{-1} = H
    inv_sqrt_H = [1.0 / _elementwise_sqrt(h, eps) for h in diag_H]  # Σ^{1/2}
    sqrt_H     = [_elementwise_sqrt(h, eps) for h in diag_H]        # Σ^{-1/2}
    H_diag     = diag_H                                             # Σ^{-1}

    # 4) n-SPSA 누적
    g_sum   = _zeros_like_params(params)
    Hp_sum  = _zeros_like_params(params)  # diag Σ'의 누적(표본 평균)
    deltaL_acc = 0.0

    for _ in range(n):
        # 4.1) u ~ N(0, I), δ = μ Σ^{1/2} u = μ * (1/sqrt(H)) ⊙ u
        u = _sample_u_like(params)
        delta = _mul_lists(inv_sqrt_H, u)     # elementwise
        _apply_delta(params, delta, +mu)
        loss_pos = causal_lm_loss(model, inputs)

        _apply_delta(params, delta, -2 * mu)
        loss_neg = causal_lm_loss(model, inputs)

        # 원복
        _apply_delta(params, delta, +mu)

        # 4.2) SPSA gradient 추정
        # g ≈ (L(θ+δ)-L(θ-δ)) / (2μ) * (Σ^{-1/2} u) = scale * (sqrt(H) ⊙ u)
        scale = float(((loss_pos - loss_neg) / (2.0 * mu)).detach().cpu())
        gu = _mul_lists(sqrt_H, u)          # Σ^{-1/2} u
        g_est = _scale_list(gu, scale)
        g_sum = [gs + ge for gs, ge in zip(g_sum, g_est)]

        # 4.3) 대각 Hessian 추정 (식 (3)의 diag)
        # ∆L = L(θ+δ) + L(θ-δ) - 2L(θ)
        deltaL = float((loss_pos + loss_neg - 2.0 * base_loss).detach().cpu())
        deltaL_acc += deltaL

        # diag(Σ^{-1/2} u u^T Σ^{-1/2}) = (Σ^{-1/2}u)^2,    diag(Σ^{-1}) = Σ^{-1} = H_diag
        Su = _mul_lists(sqrt_H, u)           # Σ^{-1/2} u
        Su2 = _sqr_list(Su)                  # (Σ^{-1/2} u)^2 (elementwise)
        # diag Σ' = 0.5 * (∆L / μ^2) * (Su2 - H_diag)
        z = 0.5 * (deltaL / (mu * mu))
        Hp = [z * (a - b) for a, b in zip(Su2, H_diag)]
        Hp_sum = [hs + hp for hs, hp in zip(Hp_sum, Hp)]

    # 5) 평균
    g_spsa = [gs / float(n) for gs in g_sum]
    diag_Sigma_prime = [hs / float(n) for hs in Hp_sum]  # Hessian (추정)의 대각

    # 6) EMA 업데이트 (양수 유지 위해 절댓값)
    diag_Sigma_prime_abs = _abs_list(diag_Sigma_prime)
    new_diag_H = _ema_update(diag_H, diag_Sigma_prime_abs, alpha=alpha)
    state["diag_H"] = new_diag_H

    # 7) 모델 상태 복구
    if was_training:
        model.train()

    metrics = {
        "base_loss": base_loss_val,
        "avg_deltaL": deltaL_acc / float(n),
        "mu": mu,
        "alpha": alpha,
        "n": n,
    }
    return g_spsa, state, metrics



import math, random
from typing import Iterator, List, Optional, Callable, Dict
import torch
from torch.utils.data import Sampler

