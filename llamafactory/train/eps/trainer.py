import json
import os
import math
import sys
import warnings
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union
import contextlib
import copy
import functools
import glob
import importlib.metadata
import inspect
import random
import re
import shutil
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override
from collections import Counter,  defaultdict, deque

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from .metric import ComputeAccuracy, ComputeSimilarity, eval_logit_processor
from dataclasses import dataclass
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import (
        PreTrainedTokenizer, 
        ProcessorMixin, 
        DataCollatorWithPadding,
        Seq2SeqTrainingArguments
    )
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments
    
from torch.utils.data.sampler import RandomSampler, Sampler
from torch.utils.data import DataLoader, Sampler
from transformers.debug_utils import DebugOption
import torch.nn.functional as F
from typing import List, Dict, Optional, Iterator, Tuple,  Any, Callable
from torch import nn
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from .eps_utils import  hizoo_spsa_diag_step, QuantileState, EpsilonPolicy
from .samplers import QuantileCurriculumSampler, RandomSamplerWithEpsilon, QuantileSamplerWithEpsilon, schedule_easy_to_hard, schedule_anti_curriculum

from accelerate.utils import (
    AutocastKwargs,
    DistributedDataParallelKwargs,
    DistributedType,
    load_fsdp_model,
    load_fsdp_optimizer,
    save_fsdp_model,
    save_fsdp_optimizer,
)

from transformers.trainer_pt_utils import (
    get_model_param_count,
)
from transformers.trainer_callback import (
    ExportableState,
    TrainerState,
)
from transformers.trainer_utils import (
    TrainOutput,
    speed_metrics,
)
from transformers.utils import (
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
)

logger = logging.get_logger(__name__)



# curriculum_sampler.py
import random
from typing import Iterator, List, Dict, Optional
from collections import defaultdict

import torch
from torch.utils.data import Sampler


from typing import Iterator, List, Dict
from collections import defaultdict
import random
from torch.utils.data import Sampler


import torch
def my_logger(info: dict):
    # 단순 출력
    print("LOG:", info)

# ---------- Configs ----------
@dataclass
class ScoreCfg:
    mode: str = "grid"   # "weighted" or "grid"
    # weighted settings
    use_loss: bool = True
    use_sharpness: bool = False
    use_grad_norm: bool = False
    w_loss: float = 1.0
    w_sharp: float = 0.0
    w_grad: float = 1.0
    norm: str = "zscore"     # "zscore" or "mad"

@dataclass
class GridCfg:
    use_loss: bool = True
    use_sharpness: bool = True
    use_grad_norm: bool = False  # 3축까지 가능하지만 2축부터 권장
    # LUTs are scales multiplied by base_rho
    lut_2d: list | None = None   # 3x3
    lut_3d: list | None = None   # 3x3x3
    def default_lut_2d(self):
        return [
          [0.6, 0.8, 1.0],
          [0.8, 1.0, 1.2],
          [1.0, 1.2, 1.4],
        ]
    def default_lut_3d(self):
        # start conservative; can tune later
        base2d = self.default_lut_2d()
        return [base2d, base2d, base2d]  # same slab for grad_norm=L/M/H

@dataclass
class SAMCfg:
    base_rho: float = 0.05
    rho_L: float = 0.7      # weighted 모드에서만 사용 (L/M/H 스케일)
    rho_M: float = 1.0
    rho_H: float = 1.3
    rho_min_scale: float = 0.3
    rho_max_scale: float = 1.7
    probe_rho: float = 0.02 # sharpness proxy 강도
    epoch_adjust_pct: float = 0.03

    def clip_bucket_scales(self):
        self.rho_L = float(min(max(self.rho_L, self.rho_min_scale), self.rho_max_scale))
        self.rho_M = float(min(max(self.rho_M, self.rho_min_scale), self.rho_max_scale))
        self.rho_H = float(min(max(self.rho_H, self.rho_min_scale), self.rho_max_scale))


def add_index(example, idx):
    example["__index__"] = idx
    return example

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        # 상태 버퍼
        if not hasattr(self, "_loss_hist"):
            self._loss_hist = deque(maxlen=self.finetuning_args.plateau_window)
        if not hasattr(self, "_probe_steps_done"):
            self._probe_steps_done = 0
        self._sam_backup = {}
        self.optim_name = str
        self.dict_lossbycat = {}
        self.dict_lossbysubcat = {}
        self.use_sam_bycat = {}
        self.log_dict = {}
        self.args.rho_eff = self.rho_eff = self.finetuning_args.sam_rho 
        self._my_sampler = Sampler
        # sam_cfg = SAMCfg(base_rho=self.finetuning_args.sam_rho, probe_rho=0.02)
        # score_cfg = ScoreCfg(mode="weighted", use_loss=True, use_sharpness=True, use_grad_norm=True, # "grid" weighted
        #                     w_loss=1.0, w_sharp=0.0, w_grad=0.0, norm="zscore")
        # grid_cfg = GridCfg(use_loss=True, use_sharpness=False, use_grad_norm = True)
        # ## Bucheck SAM
        # self.sam = sam_cfg
        # self.score = score_cfg
        # self.grid = grid_cfg
        # self.automatic_optimization = False
        # quantiles
        # if self.score.mode == "weighted":
        #     self.q33 = P2Quantile(0.33); self.q66 = P2Quantile(0.66)
        # else:
        #     if self.grid.use_loss:
        #         self.q33_loss = P2Quantile(0.33); self.q66_loss = P2Quantile(0.66)
        #     if self.grid.use_sharpness:
        #         self.q33_shp  = P2Quantile(0.33); self.q66_shp  = P2Quantile(0.66)
        #     if self.grid.use_grad_norm:
        #         self.q33_gn   = P2Quantile(0.33); self.q66_gn   = P2Quantile(0.66)

        # for val tracking
        self._best_val = None

        N = len(self.train_dataset)

        # Samplers
        self.eps_policy = EpsilonPolicy(rho_base=self.finetuning_args.sam_rho, 
                                    eps_min= 0.8,
                                    eps_mid= 1.0,
                                    eps_max= 1.1,
                                    mode="bucket")
        from datasets import Dataset
        
        self.train_dataset = self.train_dataset.map(add_index, with_indices=True)
        prior_np = self.train_dataset["diff"]  # List[float], 길이 N
        priors = torch.tensor(prior_np, dtype=torch.float32, device=self.args.device)
        self.quantile_state = QuantileState(N, 
                                            beta=0.9, 
                                            init_score=1.0, 
                                            device=self.args.device, 
                                            priors=priors, 
                                            prior_mix=0.6)
        
    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        N = len(self.train_dataset)
        device = self.args.device
        if self.finetuning_args.sampler_mode =='cur':
            # (1) Loss-based curriculum (easy→hard) 전용
            self._my_sampler = QuantileCurriculumSampler(
                n_items=N, batch_size=self.args.train_batch_size, state=self.quantile_state,
                warm_batches=self.finetuning_args.warmup_batches,
                p_schedule=schedule_easy_to_hard,  # anti는 schedule_anti_curriculum
                device=device
            )
            return self._my_sampler
            
        if self.finetuning_args.sampler_mode =='anti_cur':
            # (1) Loss-based curriculum (easy→hard) 전용
            self._my_sampler = QuantileCurriculumSampler(
                n_items=N, batch_size=self.args.train_batch_size, state=self.quantile_state,
                warm_batches=self.finetuning_args.warmup_batches,
                p_schedule=schedule_anti_curriculum,  # anti는 
                device=device
            )
            return self._my_sampler
            
        elif self.finetuning_args.sampler_mode =='randomweps':

            # (2) ε만 적용되는 랜덤 샘플러
            self._my_sampler = RandomSamplerWithEpsilon(
                n_items=N, batch_size=self.args.train_batch_size, state=self.quantile_state,
                eps_policy=self.eps_policy , device=device
            )
            return self._my_sampler

        elif self.finetuning_args.sampler_mode =='quantweps':

            # (3) p_L/M/H + ε 둘 다 적용
            def log_rho(rho_eff: float, step: int):
                self.log({"rho_eff": self.rho_eff}, step=step)
            # self.args.categories = inputs["category"].detach().cpu().tolist()
            # self.args.subcategories = inputs["subcategory"].detach().cpu().tolist()
            self._my_sampler = QuantileSamplerWithEpsilon(
                n_items=N, batch_size=self.args.train_batch_size, state=self.quantile_state,
                eps_policy=self.eps_policy,
                # warm_batches=self.finetuning_args.warmup_batches, 
                p_schedule=schedule_easy_to_hard,
                # on_rho_eff=log_rho, 
                device=device,
                # log_fn=my_logger,  
            )
            return self._my_sampler
        elif self.finetuning_args.sampler_mode =='none':
            return super()._get_train_sampler(*args, **kwargs)

        elif self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)


    # ----- utils -----
    def _batch_normalize(self, x: torch.Tensor, how:str):
        if how == "mad":
            med = torch.median(x)
            mad = torch.median(torch.abs(x - med)) + 1e-12
            return (x - med) / (mad * 1.4826)
        mu = x.mean(); sd = x.std(unbiased=False) + 1e-12
        return (x - mu) / sd

    def _seq_loss_per_sample(self, logits, labels, ignore_index=-100):
        
        # logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        # labels = inputs["labels"] if isinstance(inputs, dict) else inputs.labels
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        # labels: [B, T], logits: [B, T, V]
        B, Tm1, V = shift_logits.shape  # Tm1 = T-1

        # 잘못된(평탄화된) 버전에서 다시 복원:
        flat_logits = shift_logits.reshape(-1, V)           # [(B*Tm1), V]
        flat_labels = shift_labels.reshape(-1)              # [(B*Tm1)]
        flat_loss   = F.cross_entropy(flat_logits, flat_labels,
                                    ignore_index=IGNORE_INDEX, reduction="none")  # [(B*Tm1)]
        per_token_loss = flat_loss.view(B, Tm1)             # [B, T-1] 로 복원

        mask = (shift_labels != IGNORE_INDEX).float()       # [B, T-1]
        per_sample_loss = (per_token_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
        # self.args.losses = per_sample_loss.detach().cpu().tolist()
        
        return per_sample_loss


    def _sharpness_proxy_per_sample(self, batch, seq_loss_baseline:torch.Tensor):

        hizoo_state = {"diag_H": None}
        g_est, hizoo_state, m = hizoo_spsa_diag_step(
            self.model,
            batch,
            hizoo_state,
            mu=getattr(self.finetuning_args, "hizoo_mu", 2e-2),
            alpha=getattr(self.finetuning_args, "hizoo_alpha", 5e-6),
            n=getattr(self.finetuning_args, "train_batch_size", 1),
            only_lora=True,                 # LoRA 텐서만
            # base_loss=seq_loss_baseline,                 # 방금 구한 L(θ) 재사용
        )
        # 게이트: plateau or small grad or flat curvature → local minimum
        # hessian diagonal stats
        H_vals = torch.cat([h.flatten() for h in hizoo_state["diag_H"]])
        self.args.hessian =  H_vals.max().item()
        return H_vals.max().detach().item()
        
    # ----- scoring (weighted) -----
    def _scores_weighted(self, batch, logits, labels, _grad_norm):
        seq_loss = self._seq_loss_per_sample(logits, labels).detach()  # [B]
        # grads needed for grad_norm & sharpness: assume grads already computed
        pieces = []

        if self.score.use_loss:
            pieces.append(("loss", self._batch_normalize(seq_loss, self.score.norm)))
        # sharp = None
        if self.score.use_sharpness:
            sharp = self._sharpness_proxy_per_sample(batch, seq_loss)
            pieces.append(("sharp", self._batch_normalize(sharp, self.score.norm)))


        if self.score.use_grad_norm:
            gn_b = torch.full_like(seq_loss, _grad_norm)
            pieces.append(("grad", self._batch_normalize(gn_b, self.score.norm)))

        # weighted sum
        total = torch.zeros_like(seq_loss)
        
        for name, v in pieces:
            w = {"loss": self.score.w_loss, "sharp": self.score.w_sharp, "grad": self.score.w_grad}[name]
            total = total + w * v
        return total, {"loss": seq_loss, "sharp": sharp, "grad": gn_b}

    # ----- bucketed rho (weighted) -----
    def _rho_from_weighted_score(self, scores: torch.Tensor):
        v33 = self.q33.value(); v66 = self.q66.value()
        if v33 is None or v66 is None:
            rho_i = torch.full_like(scores, self.sam.rho_M * self.sam.base_rho)
        else:
            rho_i = torch.empty_like(scores)
            rho_i[scores < v33] = self.sam.rho_L * self.sam.base_rho
            rho_i[(scores >= v33) & (scores < v66)] = self.sam.rho_M * self.sam.base_rho
            rho_i[scores >= v66] = self.sam.rho_H * self.sam.base_rho
            print("rho_i", rho_i)
            print("scores", scores)
            print("v33, v66", v33, v66)
        return rho_i

    # ----- bucket helpers (grid) -----
    def _bucket_index(self, x, q33, q66):
        if (q33 is None) or (q66 is None): return torch.ones_like(x, dtype=torch.long)  # M
        return torch.where(x < q33, 0, torch.where(x < q66, 1, 2))  # 0:L,1:M,2:H

    def _rho_from_grid(self, batch, logits, labels, _grad_norm):
        # per-sample signals
        loss_s = self._seq_loss_per_sample(logits, labels).detach() if self.grid.use_loss else None

        sharp_s = None
        if self.grid.use_sharpness:
            sharp_s = self._sharpness_proxy_per_sample(batch, loss_s if loss_s is not None else torch.zeros_like(labels[:,0], dtype=torch.float))

        # gn = self._grad_norm().detach() if self.grid.use_grad_norm else None

        B = logits.size(0)
        # update quantiles
        if self.grid.use_loss:  self.q33_loss.update(loss_s);  self.q66_loss.update(loss_s)
        if self.grid.use_sharpness:self.q33_shp.update(sharp_s);  self.q66_shp.update(sharp_s)
        if self.grid.use_grad_norm:
            gn_b = torch.full((B,), float(_grad_norm), device=logits.device)
            self.q33_gn.update(gn_b); self.q66_gn.update(gn_b)

        # bucket indices
        b_loss  = self._bucket_index(loss_s,  self.q33_loss.value() if self.grid.use_loss else None,
                                               self.q66_loss.value() if self.grid.use_loss else None) if self.grid.use_loss else None
        b_shp   = self._bucket_index(sharp_s, self.q33_shp.value()  if self.grid.use_sharpness else None,
                                               self.q66_shp.value()  if self.grid.use_sharpness else None) if self.grid.use_sharpness else None
        b_gn    = None
        if self.grid.use_grad_norm:
            q33_gn = self.q33_gn.value(); q66_gn = self.q66_gn.value()
            b_gn = self._bucket_index(torch.full((B,), float(_grad_norm), device=logits.device), q33_gn, q66_gn)

        # LUT
        if self.grid.use_grad_norm:
            lut3 = self.grid.lut_3d or self.grid.default_lut_3d()
            rho_scale = torch.empty((B,), device=logits.device)
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        mask = ((b_loss==i) if b_loss is not None else torch.ones(B, dtype=torch.bool, device=logits.device)) & \
                               ((b_shp==j)  if b_shp  is not None else torch.ones(B, dtype=torch.bool, device=logits.device)) & \
                               (b_gn==k)
                        rho_scale[mask] = lut3[k][i][j]  # order: [gn][loss][sharp]
        else:
            lut2 = self.grid.lut_2d or self.grid.default_lut_2d()
            # pick two axes in priority: loss, sharpness
            if self.grid.use_loss and self.grid.use_sharpness:
                i_idx, j_idx = b_loss, b_shp
            elif self.grid.use_loss:
                i_idx, j_idx = b_loss, torch.ones_like(b_loss)  # middle for missing
            else:
                i_idx, j_idx = b_shp, torch.ones_like(b_shp)
            rho_scale = torch.empty((B,), device=logits.device)
            for i in range(3):
                for j in range(3):
                    mask = (i_idx==i) & (j_idx==j)
                    rho_scale[mask] = lut2[i][j]
        return rho_scale * self.sam.base_rho
    
    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        del inputs
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes


        return (loss, outputs) if return_outputs else loss



    @override
    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch, return_outputs=True)

        # del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            # Finally we need to normalize the loss for reporting
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False
                
            loss_clean = self.compute_loss(model, inputs)
            self.accelerator.backward(loss, **kwargs)

            # ---- 준비 ----
            batch_idx_t   = inputs["__index__"].to(self.args.device, dtype=torch.long)   # [B]
            batch_idx_list = batch_idx_t.tolist()
            # fwd_inputs    = {k: v for k, v in inputs.items() if k != "__index__"}

            # ---- 1) 1st forward: loss + logits (per-sample loss 계산용) ----
            outputs = model(**inputs)
            logits  = outputs.logits
            loss    = outputs.loss  # scalar

            # per-sample 평균 NLL (원본 forward 기준)
            seq_loss = self._seq_loss_per_sample(logits, inputs["labels"])  # [B]

            # ---- 2) 1st backward: grad 확보 (SAM/grad_norm/헤시안용) ----
            # Trainer 관례: training_step 안에서 backward 수행
            self.accelerator.backward(loss)

            # 이제 p.grad가 채워졌으니 grad-norm 계산 가능
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                total_norm_sq += g.data.norm(2).item() ** 2
            total_norm = total_norm_sq ** 0.5

            self.args.hessian = self._sharpness_proxy_per_sample(inputs, seq_loss)
            # ---- 4) ε 정책으로 ρ_eff 계산 (동적/정적) ----
                # rho_eff, stats = self.eps_policy.rho_eff_with_stats(self.quantile_state, batch_idx_t)
                # self.log({"rho_eff": rho_eff, **{f"eps/{k}": v for k, v in stats.items()}})
            # rho_eff, eps = self.eps_policy.rho_eff(self.quantile_state, batch_idx_t, return_eps=True)
            # if not self.finetuning_args.use_static:
            #     self.args.rho_eff = self.rho_eff = rho_eff
            #     self.args.kl = self._my_sampler.kl    
                
            # else:
            #     rho_eff = self.finetuning_args.sam_rho  # 고정 rho

            # 로깅용 보조 정보
            # self.args.ths = self.quantile_state.thresholds()
            # scores = self.quantile_state.score_for(batch_idx_t)
            # bucket_labels = self.quantile_state.classify_indices(batch_idx_t)

            # self.args.categories  = batch_idx_t.detach().cpu().tolist()               # 인덱스
            self.args.num_tokens  = int(inputs["attention_mask"].sum().item())        # 배치 토큰 수
            # ---- 5) SAM step: perturb -> 2nd forward/backward -> restore ----
            if self.finetuning_args.sam_rho != 0 and not self.finetuning_args.use_static:
                # (a) perturb: grad와 rho_eff 사용
                self.perturb_params(model, rho=self.finetuning_args.sam_rho)
                # self.perturb_params(model, rho=rho_eff)
                
                # (b) 2nd forward on perturbed params
                loss_perturbed = self.compute_loss(model, inputs)  # scalar
                self.restore_params(model)
                
                # --- flatness 계산 및 rho update ---
                if self.finetuning_args.dynamic_rho:
                    with torch.no_grad():
                        flatness = (loss_perturbed - loss_clean).clamp(min=1e-8).detach()
                        rho_eff = self.finetuning_args.sam_rho / (1 + 10 * flatness)
                        rho_eff = torch.clamp(rho_eff, min=1e-5, max=self.finetuning_args.sam_rho)
                        # print("here", flatness, rho_eff)
                        # 진짜 perturb → 최종 gradient 계산
                        self.perturb_params(model, rho=rho_eff)
                    loss_perturbed_after = self.compute_loss(model, inputs)
                    self.accelerator.backward(loss_perturbed_after)
                    self.restore_params(model)
                    ret_loss = loss_perturbed_after.detach()
                    self.args.rho_eff = [rho_eff.detach().cpu().item(), flatness.detach().cpu().item(), loss_clean.detach().cpu().item(), loss_perturbed.detach().cpu().item(), loss_perturbed_after.detach().cpu().item()]
                    self.args.categories = inputs['category'].float().mean().item()
                    print(self.args.rho_eff, self.args.categories)
                    # self.args.categories = [
                    #                 {
                    #                     "idx": int(c.item()),
                    #                     "loss": float(seq_loss[i].item()),
                    #                     "catogories": inputs["category"].detach().cpu().tolist()[i], 
                    #                 }
                    #                 for i, c in enumerate(batch_idx_t)
                    #             ]
                    return ret_loss
                        
                # (c) backward on perturbed loss (이게 최종 gradient)
            if self.finetuning_args.sam_rho != 0 and self.finetuning_args.use_static:
                self.perturb_params(model, rho=self.finetuning_args.sam_rho)
                loss_perturbed = self.compute_loss(model, inputs)  # scalar
                
                self.accelerator.backward(loss_perturbed)

                # (d) 원위치 복구
                self.restore_params(model)
                self.args.rho_eff = [rho_eff.detach().cpu().item(), flatness.detach().cpu().item(), loss_clean.detach().cpu().item(), loss_perturbed.detach().cpu().item(), loss_perturbed_after.detach().cpu().item()]
                self.args.categories = inputs['category'].float().mean().item()
                # (e) 로깅/리턴은 perturbed loss 기준이 일반적
                ret_loss = loss_perturbed.detach()
                
            else:
                # SAM 안 쓰면 1st backward가 이미 됐으므로 그냥 원래 loss 반환
                ret_loss = loss.detach()

            # ---- 6) QuantileState 업데이트 (원본 forward의 per-sample loss로) ----
            # perturb 전 원본 seq_loss로 업데이트하는 게 안전함
            # self.quantile_state.update_losses(batch_idx_t, seq_loss.detach())

            # # ---- 7) 진행도 스케줄 (prior mixing 등) ----
            # progress = self.state.global_step / max(1, self.num_training_steps)
            # self.quantile_state.schedule_prior_mix(t=progress, lam0=0.6, lam1=0.2, mode="cos")  # 필요 시

            if self.finetuning_args.sampler_mode =='quantweps': 
                self.args.categories = [
                    {
                        "idx": int(c.item()),
                        "bucket_label": str(bucket_labels[i]),  
                        "loss": float(seq_loss[i].item()),
                        "score": float(scores[i].item()),
                        "eps": float(eps[i].item())
                    }
                    for i, c in enumerate(batch_idx_t)
                ]
            # else: 
            #     self.args.categories = [
            #         {
            #             "idx": int(c.item()),
            #             "loss": float(seq_loss[i].item()),
            #             "score": float(scores[i].item()),
            #             "eps": float(eps[i].item())
            #         }
            #         for i, c in enumerate(batch_idx_t)
            #     ]
                
                
            return ret_loss



    # ----- epoch end: small auto-tuning -----
    def on_validation_epoch_end(self):
        cur = float(self.trainer.callback_metrics.get("val_loss", torch.tensor(float("inf"))))
        improved = (self._best_val is None) or (cur < self._best_val - 1e-8)
        self._best_val = cur
        scale = (1 + self.sam.epoch_adjust_pct) if improved else (1 - self.sam.epoch_adjust_pct)

        if self.score.mode == "weighted":
            self.sam.rho_L *= scale; self.sam.rho_M *= scale; self.sam.rho_H *= scale
            self.sam.clip_bucket_scales()
        else:
            # grid LUT 전체 스케일
            if self.grid.lut_2d is not None:
                for i in range(3):
                    for j in range(3):
                        self.grid.lut_2d[i][j] = float(min(max(self.grid.lut_2d[i][j]*scale,
                                                               self.sam.rho_min_scale),
                                                           self.sam.rho_max_scale))
            if self.grid.lut_3d is not None:
                for k in range(3):
                    for i in range(3):
                        for j in range(3):
                            self.grid.lut_3d[k][i][j] = float(min(max(self.grid.lut_3d[k][i][j]*scale,
                                                                       self.sam.rho_min_scale),
                                                                   self.sam.rho_max_scale))
        
    @torch.no_grad()
    def _rho_clip(self):
        smin = self.cfg.rho_min_scale * self.cfg.base_rho
        smax = self.cfg.rho_max_scale * self.cfg.base_rho
        self.cfg.rho_L = float(min(max(self.cfg.rho_L, smin/self.cfg.base_rho), smax/self.cfg.base_rho))
        self.cfg.rho_M = float(min(max(self.cfg.rho_M, smin/self.cfg.base_rho), smax/self.cfg.base_rho))
        self.cfg.rho_H = float(min(max(self.cfg.rho_H, smin/self.cfg.base_rho), smax/self.cfg.base_rho))

    def _score(self, loss_i, x, y):
        # loss_i: (B,) per-sample loss (already computed)
        if self.cfg.sharpness_alpha <= 0:
            return loss_i.detach()
        # sharpness proxy (가볍게: 1-step, batch-level 방향 공유)
        loss = loss_i.mean()
        g = torch.autograd.grad(loss, [p for p in self.model.parameters() if p.requires_grad], create_graph=False)
        # normalize direction
        vec = torch.cat([gi.view(-1) for gi in g])
        d = vec / (vec.norm() + 1e-12)
        # apply small probe
        vector_to_parameters(parameters_to_vector(self.model.parameters()) + self.cfg.probe_rho * d, self.model.parameters())
        out2 = self.model(x); loss2_i = torch.nn.functional.cross_entropy(out2, y, reduction='none') if loss_i.ndim==1 else loss_i  # adjust to your criterion
        vector_to_parameters(parameters_to_vector(self.model.parameters()) - self.cfg.probe_rho * d, self.model.parameters())
        sharp_i = (loss2_i - loss_i) / (self.cfg.probe_rho**2 + 1e-12)
        a = self.cfg.sharpness_alpha
        return (1-a)*loss_i.detach() + a*sharp_i.detach()

    def _bucket_rho(self, scores):
        q33 = self.q33.value()
        q66 = self.q66.value()
        # 경계가 충분히 학습되기 전에는 모두 M으로
        if q33 is None or q66 is None:
            r = torch.full_like(scores, self.cfg.rho_M * self.cfg.base_rho)
            return r
        r = torch.empty_like(scores)
        r[scores < q33] = self.cfg.rho_L * self.cfg.base_rho
        r[(scores >= q33) & (scores < q66)] = self.cfg.rho_M * self.cfg.base_rho
        r[scores >= q66] = self.cfg.rho_H * self.cfg.base_rho
        return r
    
    @torch.no_grad()
    def epoch_adjust(self, val_improved: bool, pct: float = 0.03):
        """에폭 종료 후: 검증이 좋아지면 ρ 스케일 +3%, 아니면 -3% (클리핑 포함)"""
        scale = (1 + pct) if val_improved else (1 - pct)
        self.cfg.rho_L *= scale
        self.cfg.rho_M *= scale
        self.cfg.rho_H *= scale
        self._rho_clip()
    

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        self.num_training_steps = num_training_steps

        return super().create_scheduler(num_training_steps, optimizer)



    # @override
    # def compute_loss(self, model, inputs, *args, **kwargs):
    #     return super().compute_loss(model, inputs, *args, **kwargs)
    
    @torch.no_grad()
    def perturb_params(self, model, rho: float, eps: float = 1e-12, exclude_bias_norm: bool = False):
        """
        SAM: w <- w + rho * g / ||g||
        - rho: perturb 반경
        - exclude_bias_norm: True면 bias/norm 파라미터는 norm 계산에서 제외
        """
        grads = []
        params = []

        # 1) perturb에 포함할 파라미터/그라드 수집
        for n, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if exclude_bias_norm and (n.endswith(".bias") or "norm" in n.lower() or "ln" in n.lower()):
                continue
            grads.append(p.grad.detach())
            params.append(p)

        if len(grads) == 0:
            return 0.0  # nothing to do

        # 2) global grad norm
        grad_norm = torch.norm(torch.stack([g.norm(p=2) for g in grads]), p=2)
        scale = rho / (grad_norm + eps)

        # 3) 파라미터 백업 & perturb 적용
        self._sam_backup.clear()
        for p in model.parameters():
            if not p.requires_grad or p.grad is None:
                continue
            # 백업
            self._sam_backup[p] = p.data.clone()
            # perturb 벡터 (모든 파라미터에 동일 scale 적용; ASAM이 필요하면 여기서 조정)
            e_w = p.grad.detach() * scale
            p.add_(e_w)  # w <- w + e_w

        return float(grad_norm)

    @torch.no_grad()
    def restore_params(self, model):
        """perturb 이전 상태로 복구"""
        if not self._sam_backup:
            return
        for p in model.parameters():
            if p in self._sam_backup:
                p.data.copy_(self._sam_backup[p])
        self._sam_backup.clear()
        
        
    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """

        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels



    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")


