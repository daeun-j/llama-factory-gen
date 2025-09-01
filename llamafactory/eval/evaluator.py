# Copyright 2025 the LlamaFactory team.
#
# This code is inspired by the Dan's test library.
# https://github.com/hendrycks/test/blob/master/evaluate_flan.py
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
#
# MIT License
#
# Copyright (c) 2020 Dan Hendrycks
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file

from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer
from .template import get_eval_template
from transformers import GenerationConfig
        
if TYPE_CHECKING:
    from numpy.typing import NDArray
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class Evaluator:
    def __init__(self, args: Optional[dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        self.eval_template = get_eval_template(self.eval_args.lang)
        self.choice_inputs = [self.tokenizer.encode(ch, add_special_tokens=False)[-1] for ch in CHOICES]

    @torch.inference_mode()
    def batch_inference(self, batch_input: dict[str, "torch.Tensor"]) -> list[str]:
        logits = self.model(**batch_input).logits

        if self.eval_args.task not in ['ifeval_test' ,'bbh_test']:
            lengths = torch.sum(batch_input["attention_mask"], dim=-1)
            word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
            choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
            return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]
        else:
            token_ids = torch.argmax(logits, dim=-1)  # [batch_size, seq_len]
            return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def eval(self) -> None:
        if self.eval_args.task not in ['ifeval_test' ,'bbh_test']:
            eval_task = self.eval_args.task.split("_")[0]
            eval_split = self.eval_args.task.split("_")[1]  

            import os
            mapping = cached_file(
                path_or_repo_id=os.path.join(self.eval_args.task_dir, eval_task),
                filename="mapping.json",
                cache_dir=self.model_args.cache_dir,
                token=self.model_args.hf_hub_token,
            )

            with open(mapping, encoding="utf-8") as f:
                categorys: dict[str, dict[str, str]] = json.load(f)

            category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
            pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
            results = {}
            for subject in pbar:  
                dataset = load_dataset(
                    path=os.path.join(self.eval_args.task_dir, eval_task),
                    name=subject,
                    cache_dir=self.model_args.cache_dir,
                    download_mode=self.eval_args.download_mode,
                    token=self.model_args.hf_hub_token,
                    trust_remote_code=self.model_args.trust_remote_code,
                )
                pbar.set_postfix_str(categorys[subject]["name"])
                inputs, outputs, labels = [], [], []
                for i in trange(len(dataset[eval_split]), desc="Formatting batches", position=1, leave=False):
                    support_set = (
                        dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                    )
                    messages = self.eval_template.format_example(
                        target_data=dataset[eval_split][i],
                        support_set=support_set,
                        subject_name=categorys[subject]["name"],
                    )

                    input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                    inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                    labels.append(messages[-1]["content"])

                for i in trange(
                    0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
                ):
                    batch_input = self.tokenizer.pad(
                        inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                    ).to(self.model.device)
                    preds = self.batch_inference(batch_input)
                    outputs += preds

                smooth = SmoothingFunction().method1
                print(labels, outputs)
                bleu_scores = [
                    sentence_bleu(
                        [label.split()],
                        output.split(),
                        smoothing_function=smooth
                    )
                    for label, output in zip(labels, outputs)
                ]
                # corrects = np.array(outputs) == np.array(labels)
                category_name = categorys[subject]["category"]
                category_corrects[category_name] = np.concatenate([category_corrects[category_name], bleu_scores], axis=0)
                category_corrects["Average"] = np.concatenate([category_corrects["Average"], bleu_scores], axis=0)
                
                # corrects = np.array(outputs) == np.array(labels)
                # category_name = categorys[subject]["category"]
                # category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
                # category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
                results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

            pbar.close()
            self._save_results(category_corrects, results)
        elif self.eval_args.task == 'ifeval_test':
        
            import ifeval
            category_corrects = {'ifeval': np.array([], dtype="bool")}
            results = {}
            ds = load_dataset("google/IFEval")
            # 저장할 JSONL 파일 경로
            with open("ifeval_train_split.jsonl", "w", encoding="utf-8") as f:
                for example in ds["train"]:
                    original_kwargs = example['kwargs']  # list of dicts
                    # None 값 제거
                    filtered_kwargs = [{k: v for k, v in d.items() if v is not None} for d in original_kwargs]
                    final_kwargs = filtered_kwargs
                    example['kwargs'] = final_kwargs
                    json.dump(example, f, ensure_ascii=False)
                    f.write("\n")
            # Load default dataset (English by default)
            input_examples = ifeval.get_default_dataset("en")

            # Get responses from your model (example)
            prompt_response_pairs = {}
            for ex in input_examples:
                prompt = ex.prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                output = self.model.generate(**inputs)
                response = self.tokenizer.decode(output[0], skip_special_tokens=True)
                prompt_response_pairs[prompt] = response

            evaluator = ifeval.Evaluator(ifeval.instruction_registry)
            import nltk
            nltk.download("punkt")
            import os
            os.environ["NLTK_DATA"] = '/cmlscratch/daeunj/miniconda3/envs/fedllm/nltk_data'
            # Run evaluation
            report, all_outputs = evaluator.evaluate(input_examples, prompt_response_pairs)

            # report, all_outputs = evaluator.evaluate(input_examples, responses)
            
            print("Strict prompt accuracy:", report["eval_results_strict"]["prompt_accuracy"])
            print("Loose prompt accuracy:", report["eval_results_loose"]["prompt_accuracy"])
            print("Strict instruction accuracy:", report["eval_results_strict"]["instruction_accuracy"])
            print("Loose instruction accuracy:", report["eval_results_loose"]["instruction_accuracy"])
            self._save_results(category_corrects, results, report)
            
        elif self.eval_args.task == 'bbh_test':
            eval_task = self.eval_args.task.split("_")[0]
            eval_split = self.eval_args.task.split("_")[1]  
            import os
            
            mapping = cached_file(
                path_or_repo_id=os.path.join(self.eval_args.task_dir, eval_task),
                filename="mapping.json",
                cache_dir=self.model_args.cache_dir,
                token=self.model_args.hf_hub_token,
            )
  
            with open(mapping, encoding="utf-8") as f:
                categorys: dict[str, dict[str, str]] = json.load(f)

            category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
            pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
            results = {}

            for subject in pbar:  
                dataset = load_dataset("SaylorTwift/bbh", 
                                    name=subject, 
                                    cache_dir=self.model_args.cache_dir,
                                    download_mode=self.eval_args.download_mode,
                                    token=self.model_args.hf_hub_token,
                                    trust_remote_code=self.model_args.trust_remote_code,)
                pbar.set_postfix_str(categorys[subject]["name"])
                inputs, outputs, labels = [], [], []
                for i in trange(len(dataset[eval_split]), desc="Formatting batches", position=1, leave=False):
                    support_set = (
                        dataset["test"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["test"]))))
                    )
                    messages = self.eval_template.format_example(
                        target_data=dataset[eval_split][i],
                        support_set=support_set,
                        subject_name=categorys[subject]["name"],
                    )

                    input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                    inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                    labels.append(messages[-1]["content"])

                for i in trange(
                    0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
                ):
                    batch_input = self.tokenizer.pad(
                        inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                    ).to(self.model.device)
                    preds = self.batch_inference(batch_input)
                    outputs += preds
                # BLEU 계산
                smooth = SmoothingFunction().method1
                print(labels, outputs)
                bleu_scores = [
                    sentence_bleu(
                        [label.split()],
                        output.split(),
                        smoothing_function=smooth
                    )
                    for label, output in zip(labels, outputs)
                ]
                # corrects = np.array(outputs) == np.array(labels)
                category_name = categorys[subject]["category"]
                category_corrects[category_name] = np.concatenate([category_corrects[category_name], bleu_scores], axis=0)
                category_corrects["Average"] = np.concatenate([category_corrects["Average"], bleu_scores], axis=0)
                results[subject] = {str(i): outputs[i] for i in range(len(outputs))}
            pbar.close()
            self._save_results(category_corrects, results)
            
            
    def _save_results(self, category_corrects: dict[str, "NDArray"], results: dict[str, dict[int, str]], report: dict = None) -> None:
        if report:
            strict_acc = report.get("eval_results_strict", {}).get("prompt_accuracy", None)
            loose_acc = report.get("eval_results_loose", {}).get("prompt_accuracy", None)
            # 3. accuracy 수치만 따로 JSON으로 저장 (선택적)
            accuracy_summary = {
                "strict_prompt_accuracy": strict_acc,
                "loose_prompt_accuracy": loose_acc,
                "per_category_accuracy": {
                    category: float(np.mean(corrects))
                    for category, corrects in category_corrects.items()
                    if len(corrects)
                },
            }
            if self.eval_args.save_dir is not None:
                os.makedirs(self.eval_args.save_dir, exist_ok=False)
                with open(os.path.join(self.eval_args.save_dir, "accuracy_summary.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(accuracy_summary, f, indent=2)
 
        else:
            score_info = "\n".join(
            [
                f"{category_name:>15}: {100 * np.mean(category_correct):.2f}"
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
            )
            print(score_info)
            if self.eval_args.save_dir is not None:
                os.makedirs(self.eval_args.save_dir, exist_ok=False)
                with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(results, f, indent=2)

                with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(score_info)


def run_eval() -> None:
    Evaluator().eval()
