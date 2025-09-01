# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os
from typing import Any

from transformers.trainer import TRAINER_STATE_NAME

from . import logging
from .packages import is_matplotlib_available


if is_matplotlib_available():
    import matplotlib.figure
    import matplotlib.pyplot as plt

import matplotlib.cm as cm
from collections import defaultdict

logger = logging.get_logger(__name__)

from .constants import TRAINER_LOG

def smooth(scalars: list[float]) -> list[float]:
    r"""EMA implementation according to TensorBoard."""
    if len(scalars) == 0:
        return []

    last = scalars[0]
    smoothed = []
    weight = 1.8 * (1 / (1 + math.exp(-0.05 * len(scalars))) - 0.5)  # a sigmoid function
    for next_val in scalars:
        smoothed_val = last * weight + (1 - weight) * next_val
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def gen_loss_plot(trainer_log: list[dict[str, Any]]) -> "matplotlib.figure.Figure":
    r"""Plot loss curves in LlamaBoard."""
    plt.close("all")
    plt.switch_backend("agg")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, losses = [], []
    for log in trainer_log:
        if log.get("loss", None):
            steps.append(log["current_steps"])
            losses.append(log["loss"])

    ax.plot(steps, losses, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(losses), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    return fig


def plot_loss(save_dictionary: str, keys: list[str] = ["loss"]) -> None:
    r"""Plot loss curves and saves the image."""
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, metrics = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                steps.append(data["log_history"][i]["step"])
                metrics.append(data["log_history"][i][key])

        if len(metrics) == 0:
            logger.warning_rank0(f"No metric {key} to plot.")
            continue

        plt.figure()
        plt.plot(steps, metrics, color="#1f77b4", alpha=0.4, label="original")
        plt.plot(steps, smooth(metrics), color="#1f77b4", label="smoothed")
        plt.title(f"training {key} of {save_dictionary}")
        plt.xlabel("step")
        plt.ylabel(key)
        plt.legend()
        figure_path = os.path.join(save_dictionary, "training_{}.png".format(key.replace("/", "_")))
        plt.savefig(figure_path, format="png", dpi=100)
        print("Figure saved at:", figure_path)


def plot_optim(save_dictionary: str) -> None:
    r"""Plot loss curves and saves the image."""
    plt.close("all")
    
    plt.switch_backend("agg")
        
    log_lines = []
    with open(os.path.join(save_dictionary, TRAINER_LOG), encoding="utf-8") as f:
        for line in f:
            log_lines.append(json.loads(line))
    # Parse each JSON line into a Python dictionary
    log_lines = [entry for entry in log_lines if "eval_loss" in entry]

    # Extract steps, eval_loss, and optimizer name
    steps = [entry["current_steps"] for entry in log_lines]
    eval_losses = [entry["eval_loss"] for entry in log_lines]
    optim_names = [entry["optim_name"] for entry in log_lines]

    # Unique optimizers
    optim_name_set = sorted(set(optim_names))
    colors = cm.get_cmap("Set1", len(optim_name_set))  # 색상 팔레트 설정

    # Plot
    plt.figure()

    for i, optim in enumerate(optim_name_set):
        x = [s for s, o in zip(steps, optim_names) if o == optim]
        y = [l for l, o in zip(eval_losses, optim_names) if o == optim]
        plt.scatter(x, y, label=optim, color=colors(i), alpha=0.7)

    plt.xlabel("Step")
    plt.ylabel("Eval Loss")
    plt.title("Eval Loss vs Step")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    figure_path = os.path.join(save_dictionary, "optim.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print("Figure saved at:", figure_path)
    
    
def plot_grad(save_dictionary: str, keys: list[str] = ["grad_norm"]) -> None:
    r"""Plot loss curves and saves the image."""
    plt.close("all")
    
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)

    for key in keys:
        steps, grad_norms = [], []
        for i in range(len(data["log_history"])):
            if key in data["log_history"][i]:
                # steps.append(data["log_history"][i]["step"])
                grad_norms.append(data["log_history"][i][key])

    # 플롯
    plt.figure()
    plt.hist(grad_norms, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    plt.xlabel("Gradient Norm")
    plt.ylabel("Frequency")
    plt.title("Distribution of Gradient Norm")
    plt.grid(True)
    plt.tight_layout()

    # 저장
    figure_path = os.path.join(save_dictionary, "gradient_norm_distribution.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print(f"Gradient norm distribution saved at: {figure_path}")


def plot_hessian(save_dictionary: str) -> None:
    r"""Plot hessian curves and saves the image."""
    plt.close("all")
    
    plt.switch_backend("agg")
    with open(os.path.join(save_dictionary, TRAINER_STATE_NAME), encoding="utf-8") as f:
        data = json.load(f)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    steps, hessianes = [], []
    for log in trainer_log:
        if log.get("hessian", None):
            steps.append(log["current_steps"])
            hessianes.append(log["hessian"])

    ax.plot(steps, hessianes, color="#1f77b4", alpha=0.4, label="original")
    ax.plot(steps, smooth(hessianes), color="#1f77b4", label="smoothed")
    ax.legend()
    ax.set_xlabel("step")
    ax.set_ylabel("Hessian trace")

    figure_path = os.path.join(save_dictionary, "hessian.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print("Figure saved at:", figure_path)
    
# 기존 매핑
KEY2ID = {
    "describe": 0,
    "provide": 1,
    "generate": 2,
    "create": 3,
    "write": 4,
    "name": 5,
    "rewrite": 6,
    "other": 7,
    "explain": 8,
    "find": 9,
    "identify": 10,
    "classify": 11,
    "construct": 12,
    "suggest": 13,
    "edit": 14,
    "calculate": 15,
    "make": 16,
    "summarize": 17,
    "convert": 18,
    "design": 19,
    "give": 20,
}




def plot_by_categoty(save_directory: str) -> None:
    """Plot per-category sample losses from training log."""
    # 역매핑: 숫자 → 태스크명
    ID2KEY = {v: k for k, v in KEY2ID.items()}
    plt.close("all")

    plt.switch_backend("agg")

    # 로그 읽기
    log_lines = []
    with open(os.path.join(save_directory, TRAINER_LOG), encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            # categories + losses가 있는 entry만 추출
            if "categories" in entry and "losses" in entry:
                log_lines.append(entry)

    # 카테고리별 데이터 저장
    cat_losses = defaultdict(list)  # {category: [(step, loss), ...]}
    for entry in log_lines:
        step = entry["current_steps"]
        categories = entry["categories"]
        losses = entry["losses"]
        categories = [ID2KEY.get(cat, str(cat)) for cat in categories]
        for cat, loss in zip(categories, losses):
            cat_losses[cat].append((step, loss))

    # Plot
    plt.figure(figsize=(10, 6))
    cmap = cm.get_cmap("tab20", len(cat_losses))  # 카테고리별 색상

    for i, (cat, values) in enumerate(sorted(cat_losses.items())):
        steps, losses = zip(*values)
        plt.plot(steps, losses, label=f"{cat}", color=cmap(i), marker="o")

    plt.xlabel("Step")
    plt.ylabel("sample loss")
    plt.title("sample loss by category")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show + Save
    plt.show()
    figure_path = os.path.join(save_directory, "category_losses.png")
    plt.savefig(figure_path, format="png", dpi=100)
    print("Figure saved at:", figure_path)
