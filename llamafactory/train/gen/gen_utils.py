


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

def check_minimum(dict_lossbycat, tol=1e-4):
    """
    카테고리별로 실제 최소값에 도달했는지 확인
    Args:
        dict_lossbycat: {cat: [losses]}
        tol: 기울기 절댓값이 이 값보다 작으면 flat(minimum)으로 간주
    Returns:
        {cat: "decreasing" | "flat(minimum)" | "increasing"}
    """
    # 가장 짧은 길이 찾기
    min_len = min(len(v) for v in dict_lossbycat.values() if len(v) > 0)
    if min_len < 2:  # 너무 짧으면 판단 불가
        return {cat: "not enough data" for cat in dict_lossbycat}

    window = min_len

    result = {}
    for cat, losses in dict_lossbycat.items():
        y = np.array(losses[-window:]).reshape(-1, 1)
        x = np.arange(len(y)).reshape(-1, 1)

        model = LinearRegression().fit(x, y)
        slope = model.coef_[0][0]

        if slope < -tol:
            result[cat] = False # "decreasing"
        elif slope > tol:
            result[cat] = True # "increasing"
        else:
            result[cat] = True # "flat(minimum)"
    return result


def debug_autograd(model, inputs):
    print("== grad enabled? ", torch.is_grad_enabled())

    # 1) 학습 모드 + grad 활성화
    model.train()
    torch.set_grad_enabled(True)

    # 2) labels 보장 (CausalLM 은 labels 없으면 loss가 None 이 될 수 있음)
    batch = dict(inputs)
    if "labels" not in batch or batch["labels"] is None:
        batch["labels"] = batch["input_ids"].clone()

    # all-ignore(-100) 여부 체크 (모두 무시되면 grad=0/None)
    IGNORE_INDEX = -100
    with torch.no_grad():
        lab = batch["labels"]
        valid = (lab[:, 1:] != IGNORE_INDEX).sum().item() if lab.ndim >= 2 else (lab != IGNORE_INDEX).sum().item()
    print(f"valid label count: {valid}")

    # 3) forward
    with torch.enable_grad():
        out = model(**batch)
        loss = out.loss
    print("[loss]", loss, "requires_grad:", loss.requires_grad, "grad_fn:", loss.grad_fn)

    # 4) 파라미터 수집(훈련대상만)
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print("trainable param count:", len(named_params))

    # 5) 먼저 .backward 로 연결 확인
    for _, p in named_params:
        p.grad = None
    try:
        loss.backward(retain_graph=True)
    except Exception as e:
        print("[backward error]", repr(e))
        return

    none_after_backward = [n for n, p in named_params if p.grad is None]
    print("none grads after backward:", len(none_after_backward))
    if none_after_backward:
        print("first few none:", none_after_backward[:10])

    # 6) autograd.grad 로도 시도 (명시적 리스트 + allow_unused=True)
    for _, p in named_params:
        p.grad = None
    params_only = [p for _, p in named_params]
    try:
        grads = torch.autograd.grad(
            loss, params_only, create_graph=True, retain_graph=True, allow_unused=True
        )
    except Exception as e:
        print("[autograd.grad error]", repr(e))
        return

    none_in_grads = sum(g is None for g in grads)
    print("autograd.grad none count:", none_in_grads)
    if none_in_grads:
        # 어떤 텐서가 None인지 이름과 함께 보여주기
        for (n, _), g in zip(named_params, grads):
            if g is None:
                print(" -> None grad at:", n)

    return loss, grads

# --- LoRA 파라미터 수집 (이름 기반) ---
LORA_NAME_KEYS = [
    ".lora_A.", ".lora_B.", ".lora_embedding_A.", ".lora_embedding_B.", ".lora_magnitude_vector."
]

def collect_lora_params_by_name(model: torch.nn.Module):
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad and any(k in name for k in LORA_NAME_KEYS):
            params.append(p)
    # 백업: 못 찾으면 학습 가능한 전체 파라미터
    if not params:
        params = [p for _, p in model.named_parameters() if p.requires_grad]
    return params

def _grad_or_zeros(loss, params, create_graph=True, retain_graph=True):
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=retain_graph, allow_unused=True
    )
    # None -> 0
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

def _rand_rademacher_like(params):
    vs = []
    for p in params:
        v = torch.empty_like(p).bernoulli_(0.5).mul_(2).add_(-1)  # {+1,-1}
        vs.append(v)
    return vs

def _dot_list(a_list, b_list):
    return sum((a*b).sum() for a, b in zip(a_list, b_list))


    
# -------- 파라미터 관련 --------
def _trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]

def _flatten(params_or_grads: List[torch.Tensor]) -> torch.Tensor:
    """리스트를 하나의 flat vector로."""
    vecs = []
    for t in params_or_grads:
        if t is None: continue
        vecs.append(t.reshape(-1))
    if not vecs:
        return torch.tensor(0.0, device=params_or_grads[0].device)
    return torch.cat(vecs)

def _sqnorm(grads: List[torch.Tensor]) -> torch.Tensor:
    v = _flatten(grads)
    return (v @ v)

def _dot(a: List[torch.Tensor], b: List[torch.Tensor]) -> torch.Tensor:
    return (_flatten(a) @ _flatten(b))

def _normalize(grads: List[torch.Tensor], eps: float = 1e-12) -> List[torch.Tensor]:
    n2 = _sqnorm(grads)
    n = torch.sqrt(n2 + eps)
    return [(g / n) if g is not None else None for g in grads]

def _zeros_like_params(params: List[nn.Parameter]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) for p in params]

# -------- 파라미터 perturb/restore --------
@torch.no_grad()
def _add_scaled_(params: List[nn.Parameter], vec_list: List[torch.Tensor], alpha: float):
    for p, v in zip(params, vec_list):
        if v is None: continue
        p.add_(alpha * v)

def _apply_perturb(params: List[nn.Parameter], direction: List[torch.Tensor], rho: float):
    with torch.no_grad():
        for p, d in zip(params, direction):
            if d is None: continue
            p.add_(rho * d)
    return params

def _restore(params: List[nn.Parameter], backup: List[torch.Tensor]):
    with torch.no_grad():
        for p, b in zip(params, backup):
            p.copy_(b)

def _backup_params(params: List[nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]

@torch.no_grad()
def load_params_into_model(model, params_new: List[torch.Tensor]):
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            p.copy_(params_new[i])   # in-place update
            i += 1

# -------- grads --------
def _grad_wrt_params(loss: torch.Tensor, params: List[nn.Parameter], create_graph=False, retain_graph=True, allow_unused=True) -> List[torch.Tensor]:
    grads = torch.autograd.grad(
        loss, params, create_graph=create_graph, retain_graph=retain_graph, allow_unused=allow_unused
    )
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

# -------- loss --------
def _ce_loss_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )

@torch.no_grad()
def _labels_from_inputs(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "labels" in inputs and inputs["labels"] is not None:
        return inputs["labels"]
    return inputs["input_ids"]  # HF 내부처럼 시프트로 loss 계산


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

# =========================
# 1) LoRA grads 추출 (class 밖)
# =========================

def debug_hvp_state(model, loss, params):
    print("[loss] requires_grad:", getattr(loss, "requires_grad", None))
    print("[loss] grad_fn:", getattr(loss, "grad_fn", None))

    # params 요약
    req_flags = [p.requires_grad for p in params]
    print("[params] count:", len(params),
          " requires_grad True:", sum(bool(x) for x in req_flags))

    # 1차 grad (그래프 유지!)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
    none_cnt = sum(g is None for g in grads)
    print("[grads] none_count:", none_cnt)

    # 임의 벡터 v
    v = []
    for p, g in zip(params, grads):
        if g is None:
            # 0으로 대체 (shape는 p 기준)
            v.append(torch.zeros_like(p))
        else:
            v.append(torch.randn_like(g))

    # g·v (여기 결과가 반드시 requires_grad=True)
    terms = []
    for g, v_ in zip(grads, v):
        if g is None: 
            continue
        terms.append((g * v_).sum())
    if len(terms) == 0:
        print("[gv] all grads were None → cannot proceed")
        return
    gv = sum(terms)

    print("[gv] requires_grad:", gv.requires_grad, " grad_fn:", gv.grad_fn)

    try:
        hv = torch.autograd.grad(gv, params, retain_graph=True, allow_unused=True)
        print("[hv] computed. any None:", any(h is None for h in hv))
    except RuntimeError as e:
        print("[hv] RuntimeError:", e)
        
        
def grads_wrt_lora_params(
    model: nn.Module,
    compute_loss_fn,                       # e.g., trainer.compute_loss
    inputs: Dict[str, torch.Tensor],
    create_graph: bool = False,
    retain_graph: bool = True,
    allow_unused: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[nn.Parameter]]:
    """
    returns: (loss, grads, params)
      - params: 학습대상(LoRA) 파라미터 리스트
      - grads : 각 param에 대한 d(loss)/d(param); None -> zeros_like 로 치환
    """
    model.train()
    enable_adapters_if_any(model)

    # 1) LoRA params (없으면 requires_grad=True 전체)
    params = collect_lora_params_by_name(model)
    if not params:
        raise RuntimeError("No trainable LoRA params found (check requires_grad & names).")

    # 2) loss (HF compute_loss 그대로 받음)
    loss = compute_loss_fn(model, inputs)

    # 3) grads
    grads = torch.autograd.grad(
        loss, params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
    return loss, grads, params

def collect_lora_params_by_name(model, keyword: str = "lora") -> list:
    """
    모델 안에서 이름에 특정 키워드(기본: "lora")가 들어간 파라미터만 뽑아온다.
    LoRA 모듈 파라미터만 학습 대상이라고 가정할 때 사용.

    Args:
        model   : nn.Module
        keyword : str, default="lora"

    Returns:
        params : List of nn.Parameter
    """
    params = []
    for name, p in model.named_parameters():
        if p.requires_grad and keyword in name.lower():
            params.append(p)
    return params

# ======================================
# 2) (옵션) backward 경유 grads (원하면 사용)
# ======================================
def get_loss_and_grads(
    model: nn.Module,
    compute_loss_fn,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, List[nn.Parameter], List[torch.Tensor]]:
    """
    backward(retain_graph=True) 경유로 grads 수집 (필요 시 사용).
    """
    model.train()
    enable_adapters_if_any(model)

    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None

    loss = compute_loss_fn(model, inputs)
    loss.backward(retain_graph=True)

    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    return loss, params, grads


# ======================================
# 3) τ 측정 (δ_logit / δ_func / g_SP ≈ g_SAM)
# ======================================
def measure_tau_torch(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    compute_loss_fn,                       # e.g., trainer.compute_loss
    rho: float = 0.05,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    반환:
      {
        "loss": float,
        "tau_logit": float,
        "tau_func": float,
        "tau_cross": float,
        "tau_sum": float
      }
    주의:
    - labels가 전부 ignore_index이면 dL/dlogits가 0이 될 수 있으니 유효 마스크로 안전 처리
    - LoRA 파라미터만 perturb
    """
    # 0) 준비: LoRA 파라미터
    model.train()
    enable_adapters_if_any(model)
    params = collect_lora_params_by_name(model)
    if not params:
        raise RuntimeError("No trainable LoRA params found.")

    # 1) base loss & grad g_θ  (정규화 방향만 필요)
    loss, g_theta, params = grads_wrt_lora_params(
        model, compute_loss_fn, inputs,
        create_graph=False, retain_graph=True, allow_unused=True
    )
    if float(_sqnorm(g_theta).detach().cpu()) == 0.0:
        # 의미있는 tau 계산 불가
        return {
            "loss": float(loss.detach().cpu()),
            "tau_logit": float("nan"),
            "tau_func": float("nan"),
            "tau_cross": float("nan"),
            "tau_sum": float("nan"),
        }

    g_hat = _normalize(g_theta)

    # 2) θ' = θ + ρ ĝ  (in-place perturb, no_grad)
    backup = _backup_params(params)
    _apply_perturb(params, g_hat, rho)

    # ---- (a) δ_logit ----
    # perturbed forward
    outputs_pert = model(**inputs)
    logits_pert = outputs_pert.logits  # [B,T,V]

    # 안전한 CE 구성 (유효 토큰만)
    labels = inputs.get("labels")
    if labels is None:
        labels = inputs["input_ids"].clone()

    shift_logits = logits_pert[:, :-1, :].contiguous()     # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()              # [B, T-1]
    B, Tm1, V = shift_logits.shape
    flat_logits = shift_logits.view(-1, V)                 # [(B*(T-1)), V]
    flat_labels = shift_labels.view(-1)                    # [(B*(T-1))]

    valid = flat_labels != ignore_index
    if not valid.any():
        # 전부 ignore_index → dL/dlogits가 0 → tau 의미 없음
        _restore(params, backup)
        return {
            "loss": float(loss.detach().cpu()),
            "tau_logit": float("nan"),
            "tau_func": float("nan"),
            "tau_cross": float("nan"),
            "tau_sum": float("nan"),
        }

    loss_pert_logits = F.cross_entropy(flat_logits[valid], flat_labels[valid], reduction="mean")
    dL_dlogits = torch.autograd.grad(
        loss_pert_logits, logits_pert, retain_graph=True, allow_unused=False, create_graph=False
    )[0]  # [B,T,V]

    # VJP: δ_logit = (∂logits/∂θ)^T (dL/dlogits)
    s = (logits_pert * dL_dlogits).sum()  # <logits, dL/dlogits>
    delta_logit = torch.autograd.grad(
        s, params, retain_graph=True, allow_unused=True, create_graph=False
    )
    delta_logit = [dl if dl is not None else torch.zeros_like(p) for dl, p in zip(delta_logit, params)]

    # ---- (b) g_SAM(θ) ≈ ∇_θ L(θ+ρĝ) ----
    loss_pert = compute_loss_fn(model, inputs)  # θ'에서 loss
    g_sam = torch.autograd.grad(
        loss_pert, params, create_graph=False, retain_graph=False, allow_unused=True
    )
    g_sam = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_sam, params)]

    # 원복
    _restore(params, backup)

    # δ_func = g_SAM - δ_logit, g_SP ≈ g_SAM
    delta_func = [gs - dl for gs, dl in zip(g_sam, delta_logit)]
    g_sp = g_sam

    # 3) τ 계산
    eps = 1e-12
    gsp_n2 = _sqnorm(g_sp) + eps
    logit_n2 = _sqnorm(delta_logit)
    func_n2  = _sqnorm(delta_func)
    cross    = _dot(delta_logit, delta_func)

    tau_logit = float((logit_n2 / gsp_n2).detach().cpu())
    tau_func  = float((func_n2  / gsp_n2).detach().cpu())
    tau_cross = float((2.0 * cross / gsp_n2).detach().cpu())
    tau_sum   = tau_logit + tau_func + tau_cross

    return {
        "loss": float(loss.detach().cpu()),
        "tau_logit": tau_logit,
        "tau_func": tau_func,
        "tau_cross": tau_cross,
        "tau_sum": tau_sum,
    }
    
    
# ========== HiZOO (SPSA + Diag-Hessian EMA) for PyTorch ==========
# 목적:
#  - d차원 파라미터(LoRA 등 학습 텐서)에 대해 O(d) 메모리로
#    * n-SPSA gradient 추정
#    * 식(3) 기반 대각 Hessian 추정 (EMA)
#
# 사용처:
#  - HF Trainer의 training_step 안에서 backward 직전에 호출
#  - zeroth-order이므로 autograd로 2계미분을 만들지 않음
#
# 주의:
#  - 여기서의 "diag_H"는 ∇²L(θ)의 대각 추정치
#  - Σ = H^{-1} (근사)로 두고, perturb는 Σ^{1/2} = 1/sqrt(diag_H + eps)로 스케일
#  - 식(3)에서 Σ^{-1/2} = sqrt(diag_H), Σ^{-1} = diag_H 를 사용

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
    only_lora: bool = False,
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

