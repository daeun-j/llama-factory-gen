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
from .gen_utils import  hizoo_spsa_diag_step, check_minimum, _rand_rademacher_like, collect_lora_params_by_name, _grad_or_zeros, _dot_list, debug_hvp_state, debug_autograd

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
from dataclasses import dataclass

class Warmup_center_outer_Sampler(Sampler[int]):
    def __init__(self, dataset, batch_size: int, warmup_batches: int = 5, seed: int | None = 42):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.warmup_batches = int(warmup_batches)
        self.seed = seed
        self.rng = random.Random(seed)  # ❗️한 번만 초기화

        # category -> indices
        cat2idx: Dict[int, List[int]] = defaultdict(list)
        try:
            cats = list(self.dataset["subcategory"])
        except Exception:
            cats = [int(self.dataset[i]["subcategory"]) for i in range(len(self.dataset))]
        for idx, cat in enumerate(cats):
            cat2idx[int(cat)].append(idx)

        self.cat2idx = {c: v[:] for c, v in cat2idx.items()}
        self.round_order = sorted(self.cat2idx.keys())
        self.ptr = {c: 0 for c in self.round_order}
        # 내부 셔플(재현성 위해 커스텀 셔플 사용)
        for c in self.round_order:
            self._shuffle_inplace(self.cat2idx[c])

        self.locked_category: int | None = None
        self.emitted_batches = 0  # 배치 단위 카운팅

    def __len__(self):
        return sum(len(v) for v in self.cat2idx.values())

    # ---------- 내부 유틸 ----------
    def _shuffle_inplace(self, lst: List[int]):
        # random.shuffle(lst)는 전역 RNG를 쓰므로 재현성 위해 Fisher–Yates로 self.rng 사용
        for i in range(len(lst) - 1, 0, -1):
            j = int(self.rng.random() * (i + 1))
            lst[i], lst[j] = lst[j], lst[i]

    def _cats_with_remaining(self) -> List[int]:
        return [c for c in self.round_order if self.ptr[c] < len(self.cat2idx[c])]
    
    def set_epoch(self, epoch: int):
        if self.seed is not None:
            # 에폭마다 seed 달리 해서 섞이게
            self.rng.seed(self.seed + epoch)
        self.emitted_batches = 0
        self.ptr = {c: 0 for c in self.round_order}
        for c in self.round_order:
            self._shuffle_inplace(self.cat2idx[c])
        self.locked_category = None
    # ---------- 메인 ----------
    def __iter__(self) -> Iterator[int]:
        self.locked_category = None

        while True:
            # 남은 데이터가 없으면 종료
            if not self._cats_with_remaining():
                break
            if self.emitted_batches < self.warmup_batches:
                cats = self.round_order[:]
                self.rng.shuffle(cats)

                batch = []
                while len(batch) < self.batch_size:
                    any_added = False
                    for c in cats:
                        p = self.ptr[c]
                        buf = self.cat2idx[c]
                        if p < len(buf):
                            batch.append(buf[p])
                            self.ptr[c] = p + 1
                            any_added = True
                            if len(batch) == self.batch_size:
                                # print('warmup', batch)
                                break
                    if not any_added:
                        break

                if not batch:
                    break

                for i in batch:
                    yield i
                self.emitted_batches += 1
 

            else:
                candidates = [c for c in self.round_order if self.ptr[c] < len(self.cat2idx[c])]
                if not candidates:
                    break

                c = self.rng.choice(candidates)  # 매 배치마다 새로 뽑음
                buf = self.cat2idx[c]

                # 포인터 이후 tail 셔플(선택적)
                if self.ptr[c] < len(buf):
                    tail = buf[self.ptr[c]:]
                    self._shuffle_inplace(tail)
                    buf[self.ptr[c]:] = tail

                # 해당 카테고리로만 배치 채우기
                batch: List[int] = []
                while len(batch) < self.batch_size and self.ptr[c] < len(buf):
                    batch.append(buf[self.ptr[c]])
                    self.ptr[c] += 1

                if not batch:
                    # 방어적으로 다음 루프로
                    continue

                # yield는 Sampler 규약상 인덱스 단위
                for i in batch:
                    yield i
                self.emitted_batches += 1



class WarmupBalancedThenSingleSampler(Sampler[int]):
    def __init__(self, dataset, batch_size: int, warmup_batches: int = 5, seed: int | None = 42):
        super().__init__(data_source=None)
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.warmup_batches = int(warmup_batches)
        self.seed = seed
        self.rng = random.Random(seed)  # ❗️한 번만 초기화

        # category -> indices
        cat2idx: Dict[int, List[int]] = defaultdict(list)
        try:
            cats = list(self.dataset["category"])
        except Exception:
            cats = [int(self.dataset[i]["category"]) for i in range(len(self.dataset))]
        for idx, cat in enumerate(cats):
            cat2idx[int(cat)].append(idx)

        self.cat2idx = {c: v[:] for c, v in cat2idx.items()}
        self.round_order = sorted(self.cat2idx.keys())
        self.ptr = {c: 0 for c in self.round_order}
        # 내부 셔플(재현성 위해 커스텀 셔플 사용)
        for c in self.round_order:
            self._shuffle_inplace(self.cat2idx[c])

        self.locked_category: int | None = None
        self.emitted_batches = 0  # 배치 단위 카운팅

    def __len__(self):
        return sum(len(v) for v in self.cat2idx.values())

    # ---------- 내부 유틸 ----------
    def _shuffle_inplace(self, lst: List[int]):
        # random.shuffle(lst)는 전역 RNG를 쓰므로 재현성 위해 Fisher–Yates로 self.rng 사용
        for i in range(len(lst) - 1, 0, -1):
            j = int(self.rng.random() * (i + 1))
            lst[i], lst[j] = lst[j], lst[i]

    def _cats_with_remaining(self) -> List[int]:
        return [c for c in self.round_order if self.ptr[c] < len(self.cat2idx[c])]
    
    def set_epoch(self, epoch: int):
        if self.seed is not None:
            # 에폭마다 seed 달리 해서 섞이게
            self.rng.seed(self.seed + epoch)
        self.emitted_batches = 0
        self.ptr = {c: 0 for c in self.round_order}
        for c in self.round_order:
            self._shuffle_inplace(self.cat2idx[c])
        self.locked_category = None
    # ---------- 메인 ----------
    def __iter__(self) -> Iterator[int]:
        self.locked_category = None

        while True:
            # 남은 데이터가 없으면 종료
            if not self._cats_with_remaining():
                break
            if self.emitted_batches < self.warmup_batches:
                cats = self.round_order[:]
                self.rng.shuffle(cats)

                batch = []
                while len(batch) < self.batch_size:
                    any_added = False
                    for c in cats:
                        p = self.ptr[c]
                        buf = self.cat2idx[c]
                        if p < len(buf):
                            batch.append(buf[p])
                            self.ptr[c] = p + 1
                            any_added = True
                            if len(batch) == self.batch_size:
                                # print('warmup', batch)
                                break
                    if not any_added:
                        break

                if not batch:
                    break

                for i in batch:
                    yield i
                self.emitted_batches += 1
 

            else:
                candidates = [c for c in self.round_order if self.ptr[c] < len(self.cat2idx[c])]
                if not candidates:
                    break

                c = self.rng.choice(candidates)  # 매 배치마다 새로 뽑음
                buf = self.cat2idx[c]

                # 포인터 이후 tail 셔플(선택적)
                if self.ptr[c] < len(buf):
                    tail = buf[self.ptr[c]:]
                    self._shuffle_inplace(tail)
                    buf[self.ptr[c]:] = tail

                # 해당 카테고리로만 배치 채우기
                batch: List[int] = []
                while len(batch) < self.batch_size and self.ptr[c] < len(buf):
                    batch.append(buf[self.ptr[c]])
                    self.ptr[c] += 1

                if not batch:
                    # 방어적으로 다음 루프로
                    continue

                # yield는 Sampler 규약상 인덱스 단위
                for i in batch:
                    yield i
                self.emitted_batches += 1

                
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
    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        # if self.finetuning_args.disable_shuffling:
        #     return torch.utils.data.SequentialSampler(self.train_dataset)
        if self.finetuning_args.use_balanced_sampler:
            # return BalancedCategorySampler(self.train_dataset, batch_size=self.args.train_batch_size)
            return  WarmupBalancedThenSingleSampler(
                        dataset=self.train_dataset,
                        batch_size=self.args.train_batch_size,
                        warmup_batches=self.finetuning_args.warmup_batches,          # 앞의 5 iteration은 balanced
                    )
        if self.finetuning_args.use_center_outer_sampler:
            # return BalancedCategorySampler(self.train_dataset, batch_size=self.args.train_batch_size)
            return  Warmup_center_outer_Sampler(
                        dataset=self.train_dataset,
                        batch_size=self.args.train_batch_size,
                        warmup_batches=self.finetuning_args.warmup_batches,          # 앞의 5 iteration은 balanced
                    )

        return super()._get_train_sampler(*args, **kwargs)
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
        return super().create_scheduler(num_training_steps, optimizer)



    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        return super().compute_loss(model, inputs, *args, **kwargs)
    
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

    def _update_and_is_plateau(self, loss_val: float, window:int=10, tol:float=1e-3):
        if not hasattr(self, "_loss_window"):
            self._loss_window = deque(maxlen=window)

        self._loss_window.append(loss_val)

        if len(self._loss_window) < window:
            return False  # 아직 충분한 데이터 없음

        # 단순 이동 평균의 변화량 비교
        diffs = np.diff(list(self._loss_window))
        avg_change = np.mean(np.abs(diffs))
        return avg_change < tol

    def _current_grad_norm(self):
        norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)  # L2 norm
                norm += param_norm.item() ** 2
        return norm ** 0.5

    def is_local_process_zero(self):
        # distributed 환경 아니면 그냥 True
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0


    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size
        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(args, train_dataloader, total_train_batch_size)

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(train_dataloader, None if epoch_based else max_steps)
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)
        _num_steps = -1
        for epoch in range(epochs_trained, num_train_epochs):
            # use_sam=False
            # use_sams=[]
            # self._probe_steps_done=0
            epoch_dataloader = train_dataloader
            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(epoch_dataloader, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = num_examples % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + 1
            if args.gradient_accumulation_steps == 1:
                total_updates -= 1
            for _ in range(total_updates):
                update_step += 1
                _num_steps += 1
                if _num_steps % 20 ==0:
                    use_sam=False
                    use_sams=[]
                    self._probe_steps_done=0
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
                batch_samples, num_items_in_batch = self.get_batch_samples(epoch_iterator, num_batches, args.device)
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(self.model, "main_input_name", "input_ids")
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(input_tokens, device=self.args.device, dtype=torch.int64)
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().cpu().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                        if self.finetuning_args.use_sam:
                            outputs = model(**inputs)
                            # --- 추가 부분: per-sample loss ---
                            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                            labels = inputs["labels"] if isinstance(inputs, dict) else inputs.labels
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
                            # self.args.losses = per_sample_loss

                            # 1st backward까지 끝났음 → 여기서 로컬 최소점 판정
                            # if self.finetuning_args.localmin_probe_steps > 0 and self._probe_steps_done < self.finetuning_args.localmin_probe_steps:
                            #     loss_val = float(tr_loss_step.detach().to("cpu"))
                            #     plateau  = self._update_and_is_plateau(loss_val)
                            #     gnorm    = self._current_grad_norm()
                            #     small_g  = (gnorm <= self.finetuning_args.grad_norm_min)
                            #     hess_flat = False
                                # if self.finetuning_args.use_hessian_probe:

                                #     hizoo_state = {"diag_H": None}
                                #     g_est, hizoo_state, m = hizoo_spsa_diag_step(
                                #         model,
                                #         inputs,
                                #         hizoo_state,
                                #         mu=getattr(self.finetuning_args, "hizoo_mu", 5e-3),
                                #         alpha=getattr(self.finetuning_args, "hizoo_alpha", 5e-6),
                                #         n=getattr(self.finetuning_args, "hizoo_n", 1),
                                #         only_lora=True,                 # LoRA 텐서만
                                #         base_loss=tr_loss_step,                 # 방금 구한 L(θ) 재사용
                                #     )
                                # # 게이트: plateau or small grad or flat curvature → local minimum
                                #     # hessian diagonal stats
                                #     H_vals = torch.cat([h.flatten() for h in hizoo_state["diag_H"]])
                                #     hess_flat  = (H_vals.max().item() <= self.finetuning_args.hess_flat_trace_max)

                            #     is_local_min = plateau or small_g or hess_flat
                            #     # print(loss_val, plateau, gnorm, small_g, hess_flat, H_vals.max().item())
                            #     self._probe_steps_done += 1
                            #     use_sam = (self.finetuning_args.use_sam and is_local_min)
                            #     use_sams.append(use_sam)
                            #     if self._probe_steps_done == self.finetuning_args.localmin_probe_steps:
                            #         _use_sam = Counter(use_sams).most_common(1)[0][0]
                            #     else:
                            #         _use_sam = None
                            
            
                            # # tr_loss_step 계산 직후, 가끔만 프로브:
                            # if step % 50 == 0:  # 예시
                            #     try:
                            #         tau = measure_tau_torch(
                            #             model=self.model,
                            #             inputs=inputs,
                            #             compute_loss_fn=self.compute_loss,
                            #             rho=getattr(self.finetuning_args, "sam_rho", 0.05),
                            #             ignore_index=-100,
                            #         )
                            #         if self.is_local_process_zero():
                            #             print(f"[tau] loss={tau['loss']:.4f} "
                            #                 f"logit={tau['tau_logit']:.3f} func={tau['tau_func']:.3f} "
                            #                 f"cross={tau['tau_cross']:.3f} sum={tau['tau_sum']:.3f}")
                            #     except RuntimeError as e:
                            #         if self.is_local_process_zero():
                            #             print(f"[tau probe skipped] {e}")
                            
                            # {3: True, 14: True, 10: True, 4: True, 13: False, 
                            # 16: False, 9: True, 19: False, 17: False, 0: True, 
                            # 6: False, 8: True, 5: True, 11: True, 7: True, 
                            # 2: True, 1: True, 20: False, 12: False, 15: False, 18: False}

                            

                            # self.args.optim_name = self.optim_name
                            # self.args.hessian =  H_vals.max().item()
                            self.args.losses = per_sample_loss.detach().cpu().tolist()
                            # losses, cats는 step마다 이미 있는 값
                            losses = per_sample_loss.detach().cpu().tolist()
                            
                            
                            if self.finetuning_args.use_balanced_sampler:
                                cats = inputs["category"].detach().cpu().tolist()
                                self.args.categories = inputs["category"].detach().cpu().tolist()
                                
                            if self.finetuning_args.use_center_outer_sampler:
                                cats = inputs["subcategory"].detach().cpu().tolist()
                                self.args.categories = inputs["category"].detach().cpu().tolist()
                                self.args.subcategories = inputs["subcategory"].detach().cpu().tolist()
                                
                            hizoo_state = {"diag_H": None}
                            g_est, hizoo_state, m = hizoo_spsa_diag_step(
                                model,
                                inputs,
                                hizoo_state,
                                mu=getattr(self.finetuning_args, "hizoo_mu", 5e-3),
                                alpha=getattr(self.finetuning_args, "hizoo_alpha", 5e-6),
                                n=getattr(self.finetuning_args, "hizoo_n", 1),
                                only_lora=True,                 # LoRA 텐서만
                                base_loss=tr_loss_step,                 # 방금 구한 L(θ) 재사용
                            )
                            H_vals = torch.cat([h.flatten() for h in hizoo_state["diag_H"]])
                            self.args.hessian =  H_vals.max().item()

                            if (step + 1)  < self.finetuning_args.warmup_batches:
                                _use_sam = False
                                if step ==0:
                                    self.dict_lossbycat = {}
                                for l, c in zip(losses, cats):
                                    if c not in self.dict_lossbycat:
                                        # 처음 등장하면 리스트로 초기화
                                        self.dict_lossbycat[c] = [l]
                                    else:
                                        # 이미 있으면 리스트 확장
                                        self.dict_lossbycat[c].extend([l])
                                # print(step, _use_sam, losses, cats)

                            else:
                                if (step + 1) == self.finetuning_args.warmup_batches:
                                    self.use_sam_bycat = check_minimum(self.dict_lossbycat, tol=1e-2)
                                    print(self.use_sam_bycat)
                                unique = next(iter(set(cats)))  # set에서 하나 꺼내기
                                _use_sam = self.use_sam_bycat[unique]
                                # print(step, _use_sam, losses, cats)
                                
                            # SAM 수행: perturb_params → 2nd backward → restore
                            if _use_sam is not None and _use_sam == True:
                                self.optim_name = 'sam' #getattr(base_optimizer, "optim_name", base_optimizer.__class__.__name__.lower())
                                self.perturb_params(model, rho=self.finetuning_args.sam_rho)
                                loss_perturbed = self.compute_loss(model, inputs)  # 다시 forward
                                self.restore_params(model)
                            else:
                                base_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
                                self.optim_name = getattr(base_optimizer, "optim_name", base_optimizer.__class__.__name__.lower())

                            self.args.optim_name = self.optim_name
                        

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                            elif self.use_apex:
                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                _grad_norm = self.accelerator.clip_grad_norm_(
                                    model.parameters(),
                                    args.max_grad_norm,
                                )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(args, self.state, self.control)

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                        self._maybe_log_save_evaluate(
                            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if self.control.should_epoch_stop or self.control.should_training_stop:
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
 