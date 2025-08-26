# DPO Authors: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn 2023
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import importlib
import json


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from datasets import Dataset
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import os
import sys
from ..utils.trainer_utils import DPODataCollatorWithPadding, pad_to_length

print("="*60)
print("[TEST] Debug logging is working!")
print("="*60)
def debug_log(func_name: str, stage: str, data: Any = None, extra_info: str = ""):
    """Enhanced debug logging function"""
    separator = "=" * 80
    print(f"\n{separator}")
    print(f"[DEBUG] Function: {func_name} | Stage: {stage}")
    if extra_info:
        print(f"[INFO] {extra_info}")
    
    if data is not None:
        if isinstance(data, dict):
            print(f"[DATA] Dictionary with keys: {list(data.keys())}")
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                elif isinstance(value, (list, tuple)):
                    print(f"  {key}: type={type(value).__name__}, length={len(value)}")
                else:
                    print(f"  {key}: type={type(value).__name__}, value={value}")
        elif hasattr(data, 'shape'):
            print(f"[DATA] Tensor: shape={data.shape}, dtype={data.dtype}")
        elif isinstance(data, (list, tuple)):
            print(f"[DATA] {type(data).__name__}: length={len(data)}")
            if len(data) > 0:
                print(f"  First element type: {type(data[0])}")
        else:
            print(f"[DATA] Type: {type(data).__name__}, Value: {data}")
    print(separator)


def is_peft_available():
    return importlib.util.find_spec("peft") is not None

if is_peft_available():
    from peft import get_peft_model#, prepare_model_for_int8_training



class KPOTrainer(Trainer):
    r"""
    Initialize DPOTrainer.

    Args:
        model (`transformers.PreTrainedModel`):
            The model to train, preferably an `AutoModelForSequenceClassification`.
        ref_model (`PreTrainedModelWrapper`):
            Hugging Face transformer model with a casual language modelling head. Used for implicit reward computation and loss.
        beta (`float`, defaults to 0.1):
            The beta factor in DPO loss. Higher beta means less divergence from the initial policy.
        args (`transformers.TrainingArguments`):
            The arguments to use for training.
        data_collator (`transformers.DataCollator`):
            The data collator to use for training. If None is specified, the default data collator (`DPODataCollatorWithPadding`) will be used
            which will pad the sequences to the maximum length of the sequences in the batch, given a dataset of paired sequences.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `0`):
            The padding value. This argument is required if you want to use the default data collator.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        train_dataset (`datasets.Dataset`):
            The dataset to use for training.
        eval_dataset (`datasets.Dataset`):
            The dataset to use for evaluation.
        tokenizer (`transformers.PreTrainedTokenizerBase`):
            The tokenizer to use for training. This argument is required if you want to use the default data collator.
        model_init (`Callable[[], transformers.PreTrainedModel]`):
            The model initializer to use for training. If None is specified, the default model initializer will be used.
        callbacks (`List[transformers.TrainerCallback]`):
            The callbacks to use for training.
        optimizers (`Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`):
            The optimizer and scheduler to use for training.
        preprocess_logits_for_metrics (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`):
            The function to use to preprocess the logits before computing the metrics.
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        peft_config (`Dict`, defaults to `None`):
            The PEFT configuration to use for training. If you pass a PEFT configuration, the model will be wrapped in a PEFT model.
    """

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        ref_model: Union[PreTrainedModel, nn.Module] = None,
        beta: float = 0.1,
        lambda_entropy: float=0.1,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        truncation_mode: str = "keep_end",
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
    ):
        debug_log("__init__", "INPUT", {
            "beta": beta,
            "lambda_entropy":lambda_entropy,
            "label_pad_token_id": label_pad_token_id,
            "padding_value": padding_value,
            "truncation_mode": truncation_mode,
            "max_length": max_length,
            "max_prompt_length": max_prompt_length,
            "peft_config": peft_config
        }, "Initializing KPOTrainer")
        
        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                #model = prepare_model_for_int8_training(model)
                print("[error] bed case")
                exit()
            model = get_peft_model(model, peft_config)

        if data_collator is None:
            if tokenizer is None:
                raise ValueError(
                    "max_length or a tokenizer must be specified when using the default DPODataCollatorWithPadding"
                )
            if max_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_length` in the DPOTrainer's init"
                    " it will be set to `512` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_length = 512
            if max_prompt_length is None:
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `max_prompt_length` in the DPOTrainer's init"
                    " it will be set to `128` by default, but you should do it yourself in the future.",
                    UserWarning,
                )
                max_prompt_length = 128

            data_collator = DPODataCollatorWithPadding(
                tokenizer,
                max_length=max_length,
                max_prompt_length=max_prompt_length,
                label_pad_token_id=label_pad_token_id,
                padding_value=padding_value,
                truncation_mode=truncation_mode,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                # warn users
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value

        self.beta = beta
        self.lambda_entropy=lambda_entropy
        self.ref_model = ref_model

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        debug_log("__init__", "BEFORE_SUPER", {
            "use_dpo_data_collator": self.use_dpo_data_collator,
            "beta": self.beta,
            "label_pad_token_id": self.label_pad_token_id,
            "padding_value": self.padding_value
        }, "About to call parent constructor")

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )

        # Since we inherit from trainer we always have access to an accelerator
        if hasattr(self, "accelerator"):
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            debug_log("__init__", "OUTPUT", None, "Successfully initialized KPOTrainer with accelerator")
        else:
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def concatenated_inputs(self, batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        debug_log("concatenated_inputs", "INPUT", batch, "Processing batch for concatenation")
        
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        
        debug_log("concatenated_inputs", "PROCESSING", {
            "chosen_length": batch["chosen_input_ids"].shape[1],
            "rejected_max_len": rejected_max_len,
            "final_max_length": max_length
        }, "Calculated max lengths")
        
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
                debug_log("concatenated_inputs", "CHOSEN_PROCESSING", {
                    "key": k,
                    "concatenated_key": concatenated_key,
                    "pad_value": pad_value,
                    "original_shape": batch[k].shape,
                    "padded_shape": concatenated_batch[concatenated_key].shape
                })
        
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                prefix = k.split("_")[0]
                concatenated_key = "concatenated" + k[len(prefix):]
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(self.accelerator.device)
                debug_log("concatenated_inputs", "REJECTED_PROCESSING", {
                    "key": k,
                    "prefix": prefix,
                    "concatenated_key": concatenated_key,
                    "pad_value": pad_value,
                    "original_shape": batch[k].shape,
                    "final_shape": concatenated_batch[concatenated_key].shape
                })
        
        debug_log("concatenated_inputs", "OUTPUT", concatenated_batch, "Final concatenated batch")
        return concatenated_batch

    def _get_train_sampler(self, *args, **kwargs) -> Optional[torch.utils.data.Sampler]:
        debug_log("_get_train_sampler", "INPUT", {"args": args, "kwargs": kwargs})
        print("use SequentialSampler")
        sampler = SequentialSampler(self.train_dataset)
        debug_log("_get_train_sampler", "OUTPUT", {"sampler_type": type(sampler).__name__})
        return sampler
    
    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: Dict[str, torch.FloatTensor],
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: Dict[str, torch.FloatTensor],
        policy_chosen_logits: torch.FloatTensor,
        policy_rejected_logits: Dict[str, torch.FloatTensor],
        batch: Dict[str, Union[List, torch.LongTensor]],
        select_k: list = None, # batchsize
        reference_free: bool = False,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Compute the DPO loss for a batch of policy and reference model log probabilities."""
        
        debug_log("dpo_loss", "INPUT", {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps_keys": list(policy_rejected_logps.keys()) if policy_rejected_logps else [],
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps_keys": list(reference_rejected_logps.keys()) if reference_rejected_logps else [],
            "select_k": select_k,
            "reference_free": reference_free,
            "beta": self.beta,
            "lambda_entropy": self.lambda_entropy
        }, "Computing DPO loss")
    
        # Existing log-ratio DPO term
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = {}
        for key in policy_rejected_logps:
            rejected_logratios[key] = policy_rejected_logps[key] - reference_rejected_logps[key]
        
        reject_num = len(rejected_logratios.keys())
        total_num = reject_num + 1
        logratios_dict = {}
        for i in range(0, total_num):
            if i == 0:
                logratios_dict[i] = chosen_logratios
            else:
                reject_key = f"rejected{i}"
                logratios_dict[i] = rejected_logratios[reject_key]
    
        debug_log("dpo_loss", "PROCESSING", {
            "chosen_logratios": chosen_logratios,
            "rejected_logratios_keys": list(rejected_logratios.keys()),
            "reject_num": reject_num,
            "total_num": total_num,
            "logratios_dict_keys": list(logratios_dict.keys())
        }, "Computed log ratios")
    
        # 🔥 Compute entropies
        chosen_entropies = self._get_batch_entropy(policy_chosen_logits, batch["chosen_labels"])
        rejected_entropies = {
            key: self._get_batch_entropy(policy_rejected_logits[key], batch[f"{key}_labels"])
            for key in policy_rejected_logits
        }
    
        # Loop over batch items
        temp_list = []
        for batch_idx in range(len(select_k)):
            max_k = select_k[batch_idx]
            batch_idx_list = []
            for i in range(0, max_k):
                # log-ratio diff (as before)
                temp = sum(torch.exp(self.beta * (logratios_dict[j][batch_idx] - logratios_dict[i][batch_idx])) for j in range(i+1, total_num))
                temp1 = -torch.log(temp)
                
                # entropy diff: H(y-)-H(y+)
                reject_key = f"rejected{i}" if i > 0 else None
                # In dpo_loss function, around the entropy difference calculation:
                if reject_key:
                    # Ensure both entropies are properly computed for the same sequence lengths
                    entropy_diff = rejected_entropies[reject_key][batch_idx] - chosen_entropies[batch_idx]
                else:
                    entropy_diff = torch.tensor(0.0, device=chosen_entropies.device)  # Ensure device consistency
                
                # combined term
                combined = self.beta * temp1 + self.lambda_entropy * entropy_diff
                temp2 = -F.logsigmoid(combined)
                batch_idx_list.append(temp2)
            temp_list.append(sum(batch_idx_list))
        losses = torch.stack(temp_list)
        
        rejected_rewards = {}
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        for key in policy_rejected_logps:
            rejected_rewards[key] = self.beta * (policy_rejected_logps[key] - reference_rejected_logps[key]).detach()
    
        debug_log("dpo_loss", "OUTPUT", {
            "losses": losses,
            "chosen_rewards": chosen_rewards,
            "rejected_rewards_keys": list(rejected_rewards.keys()),
            "losses_mean": losses.mean().item()
        }, "Final DPO loss computation")
    
        return losses, chosen_rewards, rejected_rewards, chosen_entropies, rejected_entropies

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits."""
        
        debug_log("_get_batch_logps", "INPUT", {
            "logits": logits,
            "labels": labels,
            "average_log_prob": average_log_prob,
            "label_pad_token_id": self.label_pad_token_id
        }, "Computing batch log probabilities")
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        debug_log("_get_batch_logps", "PROCESSING", {
            "labels_shape_after_shift": labels.shape,
            "logits_shape_after_shift": logits.shape,
            "loss_mask_sum": loss_mask.sum(-1),
            "per_token_logps": per_token_logps
        }, "Computed per-token log probabilities")

        if average_log_prob:
            result = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            result = (per_token_logps * loss_mask).sum(-1)
        
        debug_log("_get_batch_logps", "OUTPUT", {
            "result": result,
            "average_log_prob": average_log_prob
        }, "Final log probabilities")
        
        return result
        
    def _get_batch_entropy(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Compute per-sequence entropy H(y|x) under the model for response tokens only.
        This function calculates entropy only for the response part, excluding the query/prompt.
        """
        # Shift logits and labels to align for next-token prediction
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        
        # Create loss mask to identify valid (non-padding) tokens
        loss_mask = (labels != self.label_pad_token_id).float()
        
        # Compute probabilities and log probabilities with numerical stability
        log_probs = F.log_softmax(logits, dim=-1)  # More numerically stable
        probs = torch.exp(log_probs)  # Convert back to probabilities
        
        # Calculate entropy: H = -sum(p * log(p))
        # Using log_probs directly for better numerical stability
        entropies = -(probs * log_probs).sum(dim=-1)  # (B, T)
        
        # Apply mask to exclude padding tokens
        masked_entropies = entropies * loss_mask
        
        # Calculate average entropy per sequence (only over valid tokens)
        valid_token_counts = loss_mask.sum(dim=-1)  # Number of valid tokens per sequence
        
        # Avoid division by zero
        valid_token_counts = torch.clamp(valid_token_counts, min=1.0)
        
        # Average entropy per sequence
        seq_entropies = masked_entropies.sum(dim=-1) / valid_token_counts
        
        return seq_entropies

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, Dict[str, torch.FloatTensor], torch.FloatTensor, Dict[str, torch.FloatTensor]]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together."""
        
        debug_log("concatenated_forward", "INPUT", {
            "model_type": type(model).__name__,
            "batch_keys": list(batch.keys()),
            "chosen_input_ids_shape": batch["chosen_input_ids"].shape if "chosen_input_ids" in batch else None
        }, "Running concatenated forward pass")
        
        concatenated_batch = self.concatenated_inputs(batch)
        
        debug_log("concatenated_forward", "AFTER_CONCATENATION", {
            "concatenated_input_ids_shape": concatenated_batch["concatenated_input_ids"].shape,
            "concatenated_attention_mask_shape": concatenated_batch["concatenated_attention_mask"].shape
        }, "After input concatenation")
        
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
        
        debug_log("concatenated_forward", "AFTER_MODEL_FORWARD", {
            "all_logits_shape": all_logits.shape,
            "all_logits_dtype": all_logits.dtype
        }, "After model forward pass")
        
        all_logps = self._get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )
        
        chosen_logps = all_logps[: batch["chosen_input_ids"].shape[0]]
        step = batch["chosen_input_ids"].shape[0]
        rejected_logps = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                cur_size = batch[key].shape[0]
                start = step * (cnt - 1)
                end = start + cur_size
                rejected_logps[f"rejected{cnt}"] = all_logps[start:end]
            
                debug_log("concatenated_forward", "REJECTED_LOGPS", {
                    "key": key,
                    "slice_range": f"{start}:{end}",
                    "slice_shape": all_logps[start:end].shape,
                    "batch_shape": batch[key].shape,
                    }, f"rejected{cnt} logps slice check")
    
    
        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                cur_size = batch[key].shape[0]
                start = step * (cnt - 1)
                end = start + cur_size
                rejected_logits[f"rejected{cnt}"] = all_logits[start:end]
        
        debug_log("concatenated_forward", "OUTPUT", {
            "chosen_logps": chosen_logps,
            "rejected_logps_keys": list(rejected_logps.keys()),
            "chosen_logits": chosen_logits,
            "rejected_logits_keys": list(rejected_logits.keys()),
            "step_size": step,
            "rejected_count": cnt
        }, "Final concatenated forward results")
        
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)

    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        
        debug_log("get_batch_metrics", "INPUT", {
            "model_type": type(model).__name__,
            "batch_keys": list(batch.keys()),
            "train_eval": train_eval
        }, "Computing batch metrics")
        
        metrics = {}
        
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        
        debug_log("get_batch_metrics", "AFTER_POLICY_FORWARD", {
            "policy_chosen_logps": policy_chosen_logps,
            "policy_rejected_logps_keys": list(policy_rejected_logps.keys())
        }, "After policy model forward pass")
        
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)

        debug_log("get_batch_metrics", "AFTER_REFERENCE_FORWARD", {
            "reference_chosen_logps": reference_chosen_logps,
            "reference_rejected_logps_keys": list(reference_rejected_logps.keys())
        }, "After reference model forward pass")

        losses, chosen_rewards, rejected_rewards, chosen_entropies, rejected_entropies = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            batch=batch,
            select_k=batch['select_k']
        )    


        reward_accuracies = None
        for key in rejected_rewards:
            if reward_accuracies is None:
                reward_accuracies = (chosen_rewards > rejected_rewards[key]).float()
            else:
                reward_accuracies *= (chosen_rewards > rejected_rewards[key]).float()

        prefix = "eval_" if train_eval == "eval" else ""
        
        metrics[f"{prefix}entropy/chosen"] = chosen_entropies.mean().item()
        for key in rejected_entropies:
            metrics[f"{prefix}entropy/{key}"] = rejected_entropies[key].mean().item()
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/{key}"] = rejected_rewards[key].cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        for key in rejected_rewards:
            metrics[f"{prefix}rewards/margins-{key}"] = (chosen_rewards - rejected_rewards[key]).cpu().numpy().mean()
        for key in policy_rejected_logps:
            metrics[f"{prefix}logps/rejected-{key}"] = policy_rejected_logps[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        for key in policy_rejected_logits:
            metrics[f"{prefix}logits/rejected-{key}"] = policy_rejected_logits[key].detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()

        debug_log("get_batch_metrics", "OUTPUT", {
            "losses_mean": losses.mean().item(),
            "metrics_keys": list(metrics.keys()),
            "chosen_rewards_mean": chosen_rewards.mean().item(),
            "reward_accuracies_mean": reward_accuracies.mean().item() if reward_accuracies is not None else None
        }, "Final batch metrics")

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        debug_log("compute_loss", "INPUT", {
            "model_type": type(model).__name__,
            "inputs_keys": list(inputs.keys()),
            "return_outputs": return_outputs,
            "use_dpo_data_collator": self.use_dpo_data_collator
        }, "Computing loss")
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        debug_log("compute_loss", "OUTPUT", {
            "loss": loss.item() if hasattr(loss, 'item') else loss,
            "metrics_keys": list(metrics.keys()),
            "return_outputs": return_outputs
        }, "Loss computation complete")

        if return_outputs:
            return (loss, metrics)
        return loss

    def generate_samples_for_evaluation(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

        debug_log("generate_samples_for_evaluation", "INPUT", {
            "model_type": type(model).__name__,
            "batch_keys": list(batch.keys()),
            "prompt_input_ids_shape": batch["prompt_input_ids"].shape if "prompt_input_ids" in batch else None
        }, "Generating evaluation samples")

        policy_output = model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        reference_output = self.ref_model.generate(
            batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            max_length=self.config.max_length,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        debug_log("generate_samples_for_evaluation", "OUTPUT", {
            "policy_output_shape": policy_output.shape,
            "reference_output_shape": reference_output.shape,
            "policy_output_decoded_length": len(policy_output_decoded),
            "reference_output_decoded_length": len(reference_output_decoded)
        }, "Generated evaluation samples")

        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        debug_log("prediction_step", "INPUT", {
            "model_type": type(model).__name__,
            "inputs_keys": list(inputs.keys()),
            "prediction_loss_only": prediction_loss_only,
            "ignore_keys": ignore_keys,
            "use_dpo_data_collator": self.use_dpo_data_collator
        }, "Running prediction step")
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        debug_log("prediction_step", "PROCESSING", {
            "final_ignore_keys": ignore_keys
        }, "Determined ignore keys")

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        debug_log("prediction_step", "AFTER_METRICS", {
            "loss": loss.item() if hasattr(loss, 'item') else loss,
            "metrics_keys": list(metrics.keys()),
            "prediction_loss_only": prediction_loss_only
        }, "Computed metrics for prediction")

        if prediction_loss_only:
            debug_log("prediction_step", "OUTPUT", {
                "loss": loss.detach(),
                "logits": None,
                "labels": None
            }, "Returning loss only")
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            # "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        debug_log("prediction_step", "OUTPUT", {
            "loss": loss.detach(),
            "logits_shape": logits.shape,
            "labels_shape": labels.shape,
            "logits_dict_keys": list(logits_dict.keys())
        }, "Final prediction step results")

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        debug_log("store_metrics", "INPUT", {
            "metrics_keys": list(metrics.keys()),
            "train_eval": train_eval,
            "current_stored_metrics_keys": list(self._stored_metrics[train_eval].keys())
        }, "Storing metrics")
        
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
        
        debug_log("store_metrics", "OUTPUT", {
            "updated_stored_metrics_keys": list(self._stored_metrics[train_eval].keys()),
            "stored_metrics_lengths": {k: len(v) for k, v in self._stored_metrics[train_eval].items()}
        }, "Metrics stored successfully")

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        debug_log("log", "INPUT", {
            "logs_keys": list(logs.keys()),
            "train_eval_determination": "train" if "loss" in logs else "eval",
            "args": args,
            "kwargs": kwargs
        }, "Logging metrics")
        
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        
        debug_log("log", "PROCESSING", {
            "train_eval": train_eval,
            "stored_metrics_keys": list(self._stored_metrics[train_eval].keys())
        }, "Processing stored metrics")
        
        # Add averaged stored metrics to logs
        original_logs_keys = set(logs.keys())
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        
        debug_log("log", "AFTER_AVERAGING", {
            "original_logs_keys": list(original_logs_keys),
            "final_logs_keys": list(logs.keys()),
            "added_keys": list(set(logs.keys()) - original_logs_keys)
        }, "Added averaged metrics to logs")
        
        del self._stored_metrics[train_eval]
        
        debug_log("log", "BEFORE_SUPER", {
            "final_logs": {k: v for k, v in logs.items()}
        }, "About to call parent log method")
        
        super().log(logs, *args, **kwargs)
        
        debug_log("log", "OUTPUT", None, "Logging completed successfully")
