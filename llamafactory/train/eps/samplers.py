

import torch
from torch.utils.data import Sampler
import math, random
from typing import List, Tuple, Optional, Dict, Iterator



# ---------- p_L/M/H 스케줄 ----------
def cosine_blend(t: float, early: Tuple[float,float,float], late: Tuple[float,float,float]) -> Tuple[float,float,float]:
    # t in [0,1]
    a = 0.5 - 0.5*math.cos(math.pi*max(0.0, min(1.0, t)))
    p0 = torch.tensor(early); p1 = torch.tensor(late)
    p = (1-a)*p0 + a*p1
    p = (p / p.sum()).tolist()
    return tuple(p)

def schedule_easy_to_hard(progress: float) -> Tuple[float,float,float]:
    # 초반 easy↑, 후반 균등
    return cosine_blend(progress, (0.7, 0.25, 0.05), (0.05,0.25,0.7))
    # return cosine_blend(progress, (0.45, 0.30, 0.25), (0.1, 0.4, 0.5))
    # return cosine_blend(progress, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))

def schedule_anti_curriculum(progress: float) -> Tuple[float,float,float]:
    # 초반 hard↑, 후반 균등
    return cosine_blend(progress, (0.1, 0.2, 0.7), (1/3,1/3,1/3))

from .eps_utils import QuantileState, EpsilonPolicy

class QuantileCurriculumSampler(Sampler[int]):
    """
    Loss-based curriculum(또는 Anti-curriculum) 전용:
    - p_schedule(progress)로 p_L/M/H를 받는다.
    - ε 적용은 하지 않는다.
    """
    def __init__(self,
                 n_items:int,
                 batch_size:int,
                 state: QuantileState,
                 warm_batches:int = 20,
                 p_schedule = schedule_easy_to_hard,  # 함수(progress:0~1)->(pL,pM,pH)
                 total_batches_hint:int | None = None,
                 seed: Optional[int] = 42,
                 device:str = "cpu"):
        super().__init__(data_source=None)
        self.n_items = n_items
        self.bs = batch_size
        self.state = state
        self.warm_batches = warm_batches
        self.p_schedule = p_schedule
        self.total_batches_hint = total_batches_hint  # 진행도 계산용(없으면 pool 기준)
        self.device = device
        self.rng = random.Random(seed)
        self._reset()
        
    def _reset(self):
        self.pool = torch.arange(self.n_items, device=self.device)
        self.emitted_batches = 0
        self.batch_since_refresh = 0
        self.L = self.M = self.H = torch.tensor([], dtype=torch.long, device=self.device)

    def __len__(self):
        return self.n_items

    def set_epoch(self, epoch:int):
        self._reset()

    @torch.no_grad()
    def _refresh_buckets_pool_based(self):
        """남은 pool_list 점수로 t33/t66 재산출 → 캐시에 set → L/M/H 재구성."""
        if len(self.pool_list) == 0:
            self.L, self.M, self.H = [], [], []
            self.batch_since_refresh = 0
            return

        # 1) <<풀 기반 경계 갱신>>
        self.state.refresh_thresholds_from_pool(self.pool_list)

        # 2) 캐시된 최신 경계로 분류
        inds = torch.tensor(self.pool_list, device=self.state.device, dtype=torch.long)
        scores = self.state.score_for(inds)
        t33, t66 = self.state.thresholds()

        L_mask = (scores < t33).tolist()
        H_mask = (scores >= t66).tolist()
        taken = set()

        self.L, self.H = [], []
        for idx, lm, hm in zip(self.pool_list, L_mask, H_mask):
            if lm:
                self.L.append(idx); taken.add(idx)
            elif hm:
                self.H.append(idx); taken.add(idx)
        self.M = [i for i in self.pool_list if i not in taken]

        self.batch_since_refresh = 0
        N = max(1, len(self.pool_list))  # 0으로 나누기 방지
        # 리프레시 시점이면 로그 찍기
        if (self.batch_since_refresh == 0) or (self.batch_since_refresh >= self.warm_batches):
            self._refresh_buckets()
            N = max(1, len(self.pool_list))
            print({
                "refresh/t33": float(self._t33),
                "refresh/t66": float(self._t66),
                "refresh/RL_frac": len(self.L) / N,
                "refresh/RM_frac": len(self.M) / N,
                "refresh/RH_frac": len(self.H) / N,
                "refresh/pool_size": N,
                "target/pL": pL,
                "target/pM": pM,
                "target/pH": pH,
            })
    def _draw(self, bucket: torch.Tensor, k:int) -> List[int]:
        if k <= 0 or len(bucket) == 0: return []
        if k >= len(bucket): return bucket.tolist()
        perm = torch.randperm(len(bucket), device=bucket.device)[:k]
        return bucket[perm].tolist()

    def _rebuild_pool(self, picked: List[int]):
        if not picked: return
        picked_set = set(picked)
        mask = ~torch.tensor([i in picked_set for i in self.pool.tolist()],
                             dtype=torch.bool, device=self.device)
        self.pool = self.pool[mask]
        for name in ("L","M","H"):
            arr = getattr(self, name)
            if len(arr)==0: continue
            m = ~torch.tensor([i in picked_set for i in arr.tolist()],
                              dtype=torch.bool, device=self.device)
            setattr(self, name, arr[m])

    def __iter__(self) -> Iterator[int]:
        while True:
            if len(self.pool) == 0: break
            if self.emitted_batches < self.warm_batches:
                # 워밍업: 균등 랜덤
                bs = min(self.bs, len(self.pool))
                perm = torch.randperm(len(self.pool), device=self.device)[:bs]
                batch = self.pool[perm].tolist()
                self._rebuild_pool(batch)
                self.emitted_batches += 1
                for i in batch: yield i
                continue
            # 리프레시
            if (self.batch_since_refresh == 0) or (self.batch_since_refresh >= self.warm_batches):
                self._refresh_buckets()

            # 진행도 추정
            if self.total_batches_hint is None:
                progress = self.emitted_batches / max(1, math.ceil(self.n_items / self.bs))
            else:
                progress = self.emitted_batches / max(1, self.total_batches_hint)
            s = pL + pM + pH
            target_L = int(round(self.bs * pL / s))
            target_H = int(round(self.bs * pH / s))
            target_M = self.bs - target_L - target_H

            pick_L = self._draw(self.L, target_L)
            pick_M = self._draw(self.M, target_M)
            left = self.bs - (len(pick_L) + len(pick_M))
            pick_H = self._draw(self.H, max(0, target_H + left))
            picked = pick_L + pick_M + pick_H
            short = self.bs - len(picked)
            if short > 0:
                picked += self._draw(self.pool, short)

            if not picked: break
            self._rebuild_pool(picked)
            self.batch_since_refresh += 1
            self.emitted_batches += 1
            for i in picked: yield i
            
            
            
class RandomSamplerWithEpsilon(Sampler[int]):
    """
    샘플링은 균등(random). ε_L/M/H만 적용해서 rho_eff를 계산(로그/사용용).
    """
    def __init__(self, n_items:int, batch_size:int,
                 state: QuantileState, eps_policy: EpsilonPolicy,
                 seed: Optional[int]=42, device:str="cpu"):
        super().__init__(data_source=None)
        self.n_items = n_items
        self.bs = batch_size
        self.state = state
        self.eps_policy = eps_policy
        self.device = device
        self.rng = random.Random(seed)
        self._reset()

    def _reset(self):
        self.pool = torch.arange(self.n_items, device=self.device)
        self.emitted_batches = 0

    def __len__(self):
        return self.n_items

    def set_epoch(self, epoch:int):
        self._reset()

    def __iter__(self) -> Iterator[int]:
        while len(self.pool) > 0:
            bs = min(self.bs, len(self.pool))
            perm = torch.randperm(len(self.pool), device=self.device)[:bs]
            batch = self.pool[perm].tolist()
            # pool 업데이트
            picked_set = set(batch)
            mask = ~torch.tensor([i in picked_set for i in self.pool.tolist()],
                                 dtype=torch.bool, device=self.device)
            self.pool = self.pool[mask]
            self.emitted_batches += 1

            for i in batch: yield i
            
import math, random
from typing import Iterator, List, Optional, Callable
import torch
from torch.utils.data import Sampler

class QuantileSamplerWithEpsilon(Sampler[int]):
    """
    - p_schedule(progress)로 p_L/M/H 목표 비율을 받음(커리큘럼/안티 가능)
    - 경계는 K-step마다 '남은 풀(pool_list)' 점수 분포로 리프레시 (pool-aware)
    - 목표 비율은 '가용량'을 고려해 할당/재분배
    - ε_L/M/H는 sampler 밖(EpsilonPolicy)에서 rho_eff 계산에 사용 (여긴 로그만)
    - 중복 샘플링 없음
    """
    def __init__(
        self,
        n_items: int,
        batch_size: int,
        state,                 # QuantileState
        eps_policy,            # EpsilonPolicy (rho_eff 계산은 밖에서)
        warmup_batches: int = 20,
        refresh_interval: int = 20,      # K-step
        p_schedule=schedule_easy_to_hard,
        total_batches_hint: Optional[int] = None,
        seed: Optional[int] = 42,
        device: str = "cpu",
        log_fn: Optional[Callable[[Dict], None]] = None,   # 로그 콜백(dict)
    ):
        super().__init__(data_source=None)
        self.n_items = int(n_items)
        self.bs = int(batch_size)
        self.state = state
        self.eps_policy = eps_policy
        self.warmup_batches = int(warmup_batches)
        self.refresh_interval = int(refresh_interval)
        self.p_schedule = p_schedule
        self.total_batches_hint = total_batches_hint
        self.device = device
        self.rng = random.Random(seed)
        self.log_fn = log_fn
        self._reset()
        self.last_batch_bucket_info: dict[int, str] = {}
        self.kl = 0
    def _reset(self):
        self.pool = torch.arange(self.n_items, device=self.device, dtype=torch.long)
        self.pool_list: List[int] = self.pool.tolist()
        self.emitted_batches = 0
        self.batch_since_refresh = 0
        self.L: List[int] = []; self.M: List[int] = []; self.H: List[int] = []

    def __len__(self):
        return self.n_items

    def set_epoch(self, epoch: int):
        self._reset()

    # --------- 내부 유틸 ---------
    def _shuffle_inplace(self, lst: List[int]):
        for i in range(len(lst) - 1, 0, -1):
            j = int(self.rng.random() * (i + 1))
            lst[i], lst[j] = lst[j], lst[i]

    def _draw(self, bucket_list: List[int], k: int) -> List[int]:
        k = min(int(k), len(bucket_list))
        if k <= 0:
            return []
        self._shuffle_inplace(bucket_list)
        pick = bucket_list[:k]
        del bucket_list[:k]
        return pick

    @torch.no_grad()
    def _classify_by_thresholds(self, idx_list: List[int]) -> Dict[int, str]:
        if not idx_list:
            return {}
        inds = torch.tensor(idx_list, device=self.state.device, dtype=torch.long)
        s = self.state.score_for(inds)
        t33, t66 = self.state.thresholds()
        out = {}
        for i, v in zip(idx_list, s.tolist()):
            out[i] = "L" if v < t33 else ("H" if v >= t66 else "M")
        return out

    # def _rebuild_pool(self):
    #     remain = self.L + self.M + self.H
    #     self.pool_list = remain[:]                      # 리스트 캐시
    #     self.pool = torch.tensor(remain, device=self.device, dtype=torch.long)
    def _rebuild_pool(self):
        if (len(self.L) + len(self.M) + len(self.H)) > 0:
            remain = self.L + self.M + self.H
            self.pool_list = remain[:]
            self.pool = torch.tensor(remain, device=self.device, dtype=torch.long)
        else:
            # ✅ L/M/H가 비어있으면 기존 pool_list 유지
            self.pool = torch.tensor(self.pool_list, device=self.device, dtype=torch.long)
    # ---- 가용량 고려 할당 + 소진율 보정 ----
    def _allocate_with_availability(self, bs: int, pL: float, pM: float, pH: float):
        RL, RM, RH = len(self.L), len(self.M), len(self.H)
        Rtot = max(1, RL + RM + RH)

        # 진행도 기반 소진율 보정(선택): 목표 p와 남은 q를 섞음
        if self.total_batches_hint is None:
            total_batches = max(1, math.ceil(self.n_items / self.bs))
        else:
            total_batches = max(1, int(self.total_batches_hint))
        progress = self.emitted_batches / total_batches
        gamma = min(0.5, progress)  # 초반 0 → 후반 0.5

        qL, qM, qH = RL / Rtot, RM / Rtot, RH / Rtot
        pL_eff = (1 - gamma) * pL + gamma * qL
        pM_eff = (1 - gamma) * pM + gamma * qM
        pH_eff = (1 - gamma) * pH + gamma * qH

        s = pL_eff + pM_eff + pH_eff + 1e-12
        tgtL = int(round(bs * pL_eff / s))
        tgtH = int(round(bs * pH_eff / s))
        tgtM = bs - tgtL - tgtH

        # 가용량 상한
        takeL = min(tgtL, RL); takeM = min(tgtM, RM); takeH = min(tgtH, RH)
        picked = takeL + takeM + takeH
        short = bs - picked
        if short > 0:
            avail = [("L", RL - takeL), ("M", RM - takeM), ("H", RH - takeH)]
            avail = [(k, v) for k, v in avail if v > 0]
            if avail:
                A = sum(v for _, v in avail)
                extra = {k: int(round(short * (v / A))) for k, v in avail}
                # 잔차 보정
                diff = short - sum(extra.values())
                if diff != 0:
                    avail_sorted = sorted(avail, key=lambda kv: kv[1], reverse=True)
                    for k, _ in avail_sorted:
                        if diff == 0: break
                        extra[k] += 1 if diff > 0 else -1
                        diff += -1 if diff > 0 else 1
                takeL += extra.get("L", 0)
                takeM += extra.get("M", 0)
                takeH += extra.get("H", 0)

        return takeL, takeM, takeH

    @torch.no_grad()
    def _refresh_buckets_pool_based(self):
        """남은 pool_list 점수로 t33/t66 재산출 → 캐시에 set → L/M/H 재구성."""
        if len(self.pool_list) == 0:
            self.L, self.M, self.H = [], [], []
            self.batch_since_refresh = 0
            return

        inds = torch.tensor(self.pool_list, device=self.state.device, dtype=torch.long)
        scores = self.state.score_for(inds)

        # 풀 기반 경계 계산 후 캐시에 저장
        t33_new, t66_new = torch.quantile(
            scores, torch.tensor([0.33, 0.66], device=scores.device)
        ).tolist()
        self.state.set_thresholds(t33_new, t66_new)

        # 캐시된 최신 경계로 마스크
        t33, t66 = self.state.thresholds()
        L_mask = (scores < t33).tolist()
        H_mask = (scores >= t66).tolist()
        L_or_H = set()

        self.L = []
        self.H = []
        for idx, lm, hm in zip(self.pool_list, L_mask, H_mask):
            if lm:
                self.L.append(idx); L_or_H.add(idx)
            elif hm:
                self.H.append(idx); L_or_H.add(idx)
        self.M = [i for i in self.pool_list if i not in L_or_H]

        self.batch_since_refresh = 0

        # (선택) 리프레시 로그
        if self.log_fn is not None:
            RL, RM, RH = len(self.L), len(self.M), len(self.H)
            self.log_fn({
                "refresh/t33": t33, "refresh/t66": t66,
                "refresh/RL": RL, "refresh/RM": RM, "refresh/RH": RH,
                "refresh/pool_size": len(self.pool_list),
            })

    # ------------- 메인 -------------
    def __iter__(self) -> Iterator[int]:
        while True:
            if len(self.pool_list) == 0:
                break

            self.last_batch_bucket_info.clear()

            # ---- 워밍업: 균등 ----
            if self.emitted_batches < self.warmup_batches:
                bs = min(self.bs, len(self.pool_list))
                self._shuffle_inplace(self.pool_list)
                batch = self.pool_list[:bs]
                del self.pool_list[:bs]
                for i in batch:
                    self.last_batch_bucket_info[i] = "W"

                # 로그
                if self.log_fn is not None:
                    self.log_fn({
                        "mode": "warmup",
                        "step": self.emitted_batches,
                        "batch_size": len(batch),
                        "pool_remain": len(self.pool_list),
                    })

            else:
                # ---- K-step 또는 저수위 리프레시 ----
                LOW_WATER = 2 * self.bs
                need_lowwater = (
                    (len(self.L) < LOW_WATER) or (len(self.M) < LOW_WATER) or (len(self.H) < LOW_WATER)
                )
                if self.batch_since_refresh >= self.refresh_interval or need_lowwater or (len(self.L)+len(self.M)+len(self.H) == 0):
                    self._refresh_buckets_pool_based()

                # ---- 진행도 & 목표 비율 ----
                if self.total_batches_hint is None:
                    total_batches = max(1, math.ceil(self.n_items / self.bs))
                else:
                    total_batches = max(1, int(self.total_batches_hint))
                progress = self.emitted_batches / total_batches
                pL, pM, pH = self.p_schedule(min(1.0, max(0.0, progress)))

                # ---- 가용량 고려 할당 ----
                takeL, takeM, takeH = self._allocate_with_availability(self.bs, pL, pM, pH)

                # ---- 추출 ----
                pick_L = self._draw(self.L, takeL)
                pick_M = self._draw(self.M, takeM)
                pick_H = self._draw(self.H, takeH)
                batch = pick_L + pick_M + pick_H

                # 표식
                for i in pick_L: self.last_batch_bucket_info[i] = "L"
                for i in pick_M: self.last_batch_bucket_info[i] = "M"
                for i in pick_H: self.last_batch_bucket_info[i] = "H"

                # ---- (일반적으로 필요 없음) 부족시 pool에서 백필 ----
                short = self.bs - len(batch)
                if short > 0 and len(self.pool_list) > 0:
                    backfill = self._draw(self.pool_list, short)
                    # 현재 경계로 재분류해서 표식/버킷에도 반영(선택)
                    bf_map = self._classify_by_thresholds(backfill)
                    for i in backfill:
                        self.last_batch_bucket_info[i] = bf_map.get(i, "M")
                        b = bf_map.get(i, "M")
                        if   b == "L": self.L.append(i)
                        elif b == "H": self.H.append(i)
                        else:          self.M.append(i)
                    batch += backfill
                    
                #남은 수량
                RL, RM, RH = len(self.L), len(self.M), len(self.H)

                nL = sum(1 for i in batch if self.last_batch_bucket_info.get(i) == "L")
                nM = sum(1 for i in batch if self.last_batch_bucket_info.get(i) == "M")
                nH = sum(1 for i in batch if self.last_batch_bucket_info.get(i) == "H")
                p_hat = torch.tensor([nL, nM, nH], dtype=torch.float32)
                p_tgt = torch.tensor([pL, pM, pH], dtype=torch.float32)
                self.kl = _kl_divergence(p_hat, p_tgt)

                # 남은 수량

                # ---- 로그: 목표 vs 실현, KL, 남은 수량 ----
                if self.log_fn is not None:
                    log_dict = {
                        "mode": "main",
                        "step": self.emitted_batches,
                        "progress": progress,
                        "batch_size": len(batch),
                        "p_target/L": float(pL), "p_target/M": float(pM), "p_target/H": float(pH),
                        "p_real/L": float(p_hat[0] / max(1, len(batch))),
                        "p_real/M": float(p_hat[1] / max(1, len(batch))),
                        "p_real/H": float(p_hat[2] / max(1, len(batch))),
                        "kl(p_real||p_target)": self.kl,
                        "remain/L": RL, "remain/M": RM, "remain/H": RH,
                        "pool_remain": len(self.pool_list),
                    }

                    self.log_fn(log_dict)

            if not batch:
                break

            # 풀 재구성(중복 방지)
            self._rebuild_pool()
            self.batch_since_refresh += 1
            self.emitted_batches += 1

            for i in batch:
                yield i
            
def _kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> float:
    """KL(p||q) with clamp."""
    p = p.clamp(min=eps); q = q.clamp(min=eps)
    p = p / p.sum(); q = q / q.sum()
    return float((p * (p / q).log()).sum().item())

def _power_mean(x: torch.Tensor, p: float) -> torch.Tensor:
    if p == 0:
        return torch.exp(torch.mean(torch.log(x.clamp_min(1e-12))))
    return (x.pow(p).mean()).pow(1.0/p)

# class QuantileSamplerWithEpsilon(Sampler[int]):
#     """
#     - p_schedule(progress)로 p_L/M/H 비율을 받음(커리큘럼/안티 가능)
#     - 동시에 ε_L/M/H 스케일로 rho_eff 계산해서 콜백으로 전달
#     """
#     def __init__(self, n_items:int, batch_size:int,
#                  state: QuantileState, eps_policy: EpsilonPolicy,
#                  warm_batches:int=20, p_schedule=schedule_easy_to_hard,
#                 #  on_rho_eff=None,  # Optional[Callable[[float,int],None]]
#                  total_batches_hint:int | None = None,
#                  seed: Optional[int]=42, device:str="cpu"):
#         super().__init__(data_source=None)
#         self.n_items = n_items
#         self.bs = batch_size
#         self.state = state
#         self.eps_policy = eps_policy
#         self.warm_batches = warm_batches
#         self.p_schedule = p_schedule
#         # self.on_rho_eff = on_rho_eff
#         self.total_batches_hint = total_batches_hint
#         self.device = device
#         self.rng = random.Random(seed)
#         self._reset()
#         self.last_batch_bucket_info = {}  # 전역 변수처럼 저장

#     def _reset(self):
#         self.pool = torch.arange(self.n_items, device=self.device)
#         self.emitted_batches = 0
#         self.batch_since_refresh = 0
#         self.L = self.M = self.H = torch.tensor([], dtype=torch.long, device=self.device)

#     def __len__(self):
#         return self.n_items

#     def set_epoch(self, epoch:int):
#         self._reset()

#     @torch.no_grad()
#     def _refresh_buckets(self):
#         if len(self.pool) == 0:
#             self.L = self.M = self.H = torch.tensor([], dtype=torch.long, device=self.device)
#             return
#         t33, t66 = self.state.thresholds(self.pool)
#         x = self.state.score_for(self.pool)
#         L_mask = x < t33
#         H_mask = x >= t66
#         M_mask = ~(L_mask | H_mask)
#         self.L = self.pool[L_mask]
#         self.M = self.pool[M_mask]
#         self.H = self.pool[H_mask]
#         self.batch_since_refresh = 0

#     def _draw(self, bucket: torch.Tensor, k:int) -> List[int]:
#         if k <= 0 or len(bucket) == 0: return []
#         if k >= len(bucket): return bucket.tolist()
#         perm = torch.randperm(len(bucket), device=bucket.device)[:k]
#         return bucket[perm].tolist()

#     def _rebuild_pool(self, picked: List[int]):
#         if not picked: return
#         picked_set = set(picked)
#         mask = ~torch.tensor([i in picked_set for i in self.pool.tolist()],
#                              dtype=torch.bool, device=self.device)
#         self.pool = self.pool[mask]

#         for name in ("L","M","H"):
#             arr = getattr(self, name)
#             if len(arr)==0: continue
#             m = ~torch.tensor([i in picked_set for i in arr.tolist()],
#                               dtype=torch.bool, device=self.device)
#             setattr(self, name, arr[m])

#     def __iter__(self) -> Iterator[int]:
#         while True:
#             if len(self.pool) == 0: break
#             # 워밍업: 균등
#             if self.emitted_batches < self.warm_batches:
#                 bs = min(self.bs, len(self.pool))
#                 perm = torch.randperm(len(self.pool), device=self.device)[:bs]
#                 batch = self.pool[perm].tolist()
#                 for i in batch:
#                     self.last_batch_bucket_info[i] = "W"
#                 # self.last_batch_bucket_info = {i: "W" for i in batch}  # 워밍업 상태는 별도 마크
#             else:
#                 # 리프레시
#                 if (self.batch_since_refresh == 0) or (self.batch_since_refresh >= self.warm_batches):
#                     self._refresh_buckets()

#                 # 진행도
#                 if self.total_batches_hint is None:
#                     progress = self.emitted_batches / max(1, math.ceil(self.n_items / self.bs))
#                 else:
#                     progress = self.emitted_batches / max(1, self.total_batches_hint)
#                 pL, pM, pH = self.p_schedule(min(1.0, max(0.0, progress)))

#                 s = pL + pM + pH
#                 target_L = int(round(self.bs * pL / s))
#                 target_H = int(round(self.bs * pH / s))
#                 target_M = self.bs - target_L - target_H
#                 pick_L = self._draw(self.L, target_L)
#                 pick_M = self._draw(self.M, target_M)
#                 left = self.bs - len(pick_L) - len(pick_M)
#                 pick_H = self._draw(self.H, max(0, left))
#                 batch = pick_L + pick_M + pick_H
#                 short = self.bs - len(batch)
#                 # ✅ 버킷 메타정보 저장
#                 for i in pick_L:
#                     self.last_batch_bucket_info[i] = "L"
#                 for i in pick_M:
#                     self.last_batch_bucket_info[i] = "M"
#                 for i in pick_H:
#                     self.last_batch_bucket_info[i] = "H"
#                 if short > 0:
#                     batch += self._draw(self.pool, short)
#                     # inds = torch.tensor(batch, device=device, dtype=torch.long)
#                     # scores = self.state.score_for(inds)   # EMA(+prior mix) 점수
#                     # t33, t66 = self.state.thresholds()    # K-step 갱신된 캐시된 경계 (계단형)
#                     # lbl = {}
#                     # for i, s in zip(idx_list, scores.tolist()):
#                     #     if s < t33:
#                     #         lbl[i] = "L"
#                     #     elif s >= t66:
#                     #         lbl[i] = "H"
#                     #     else:
#                     #         lbl[i] = "M"
#                     # return lbl
#             # ✅ 외부에서 접근 가능하도록 저장
#             if not batch: break
#             # pool/카운터 갱신
#             self._rebuild_pool(batch)
#             self.batch_since_refresh += 1
#             self.emitted_batches += 1

#             for i in batch: yield i