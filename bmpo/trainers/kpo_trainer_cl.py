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

        # Add this after your existing initialization (after self.current_step = 0 or similar)
        # Add this after your existing initialization (after self.current_step = 0 or similar)
        self.tracking_data = {
            'chosen_logps': [],
            'rejected_logps': {'rejected1': [], 'rejected2': [], 'rejected3': []},
            'chosen_entropy': [],
            'rejected_entropy': {'rejected1': [], 'rejected2': [], 'rejected3': []},
            'chosen_logratios': [],
            'rejected_logratios': {'rejected1': [], 'rejected2': [], 'rejected3': []},
            'temp1_values': [],
            'entropy_diff_values': [],
            'sample_index': []
            }
        self.total_samples_processed = 0
        
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
        # Dynamically calculate max_length from the batch
        rejected_max_len = max([batch[key].shape[1] for key in batch if key.startswith("rejected") and key.endswith("_input_ids")])
        max_length = max(batch["chosen_input_ids"].shape[1], rejected_max_len)
        
        concatenated_batch = {}
        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                pad_value = self.label_pad_token_id if "labels" in k else self.padding_value
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
                    
    
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
                
        
        return concatenated_batch
    def _get_train_sampler(self, *args, **kwargs) -> Optional[torch.utils.data.Sampler]:
        print("use SequentialSampler")
        sampler = SequentialSampler(self.train_dataset)
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
    
        
    
        # 🔥 Compute entropies
        chosen_entropies = self._get_batch_entropy(policy_chosen_logits, batch["chosen_labels"])
        rejected_entropies = {
            key: self._get_batch_entropy(policy_rejected_logits[key], batch[f"{key}_labels"])
            for key in policy_rejected_logits
        }
    
        # Loop over batch items
        temp_list = []
        temp1_tracking = []
        entropy_diff_tracking = []
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
                if i == 0:
                    # For chosen response vs all rejected responses
                    entropy_diff = torch.tensor(0.0, device=chosen_entropies.device, dtype=chosen_entropies.dtype)
                    for rej_key in rejected_entropies:
                        entropy_diff += rejected_entropies[rej_key][batch_idx] - chosen_entropies[batch_idx]
                    entropy_diff = entropy_diff / len(rejected_entropies)  # Average over rejected responses
                else:
                    # For rejected response vs chosen response
                    reject_key = f"rejected{i}"
                    entropy_diff = rejected_entropies[reject_key][batch_idx] - chosen_entropies[batch_idx]

                # Store temp1 and entropy_diff for tracking
                temp1_tracking.append(temp1.item())
                entropy_diff_tracking.append(entropy_diff.item())

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
    
        self.track_sample_metrics(policy_chosen_logps, policy_rejected_logps, chosen_entropies, rejected_entropies, 
                         chosen_logratios, rejected_logratios, temp_list, select_k)
        return losses, chosen_rewards, rejected_rewards, chosen_entropies, rejected_entropies

    def _get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits."""
        
       
        
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        
        if average_log_prob:
            result = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            result = (per_token_logps * loss_mask).sum(-1)
        
        
        return result
        
    def _get_batch_entropy(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        """
        Compute per-sequence entropy H(y|x) under the model for response tokens only.
        """
        # Shift logits and labels to align for next-token prediction
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
            
        # Ensure dimensions match after shifting
        min_length = min(labels.shape[1], logits.shape[1])
        labels = labels[:, :min_length]
        logits = logits[:, :min_length, :]
        
        # Create loss mask to identify valid (non-padding) tokens
        loss_mask = (labels != self.label_pad_token_id).float()
        
        # Compute probabilities with numerical stability
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        
        # Calculate entropy: H = -sum(p * log(p))
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
        
        
        
        concatenated_batch = self.concatenated_inputs(batch)
        
        
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
        ).logits.to(torch.float32)
     
        
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
                rejected_logps[f"rejected{cnt}"] = all_logps[step*cnt : step*(cnt+1)]
            
                
    
    
        chosen_logits = all_logits[: batch["chosen_input_ids"].shape[0]]
        rejected_logits = {}
        cnt = 0
        for key in batch:
            if key.startswith("rejected") and key.endswith("_input_ids"):
                cnt += 1
                rejected_logits[f"rejected{cnt}"] = all_logits[step*cnt : step*(cnt+1)]
        
        
        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
    def track_sample_metrics(self, chosen_logps, rejected_logps, chosen_entropies, rejected_entropies,
                            chosen_logratios, rejected_logratios, temp1_list, entropy_diff_list):
        """Track individual sample metrics for end-of-training plotting"""
        batch_size = len(chosen_logps)
    
        # Track chosen values (individual samples)
        self.tracking_data['chosen_logps'].extend(chosen_logps.detach().cpu().numpy().tolist())
        self.tracking_data['chosen_entropy'].extend(chosen_entropies.detach().cpu().numpy().tolist())
        
        # Track chosen logratios
        self.tracking_data['chosen_logratios'].extend(chosen_logratios.detach().cpu().numpy().tolist())
    
        # Track rejected values (individual samples)
        for key in rejected_logps:
            if key not in self.tracking_data['rejected_logps']:
                self.tracking_data['rejected_logps'][key] = []
                self.tracking_data['rejected_entropy'][key] = []
                self.tracking_data['rejected_logratios'][key] = []
    
            self.tracking_data['rejected_logps'][key].extend(rejected_logps[key].detach().cpu().numpy().tolist())
            self.tracking_data['rejected_entropy'][key].extend(rejected_entropies[key].detach().cpu().numpy().tolist())
            self.tracking_data['rejected_logratios'][key].extend(rejected_logratios[key].detach().cpu().numpy().tolist())
        
        # Track temp1 and entropy_diff values
        self.tracking_data['temp1_values'].extend(temp1_list)
        self.tracking_data['entropy_diff_values'].extend(entropy_diff_list)
    
        # Track sample indices
        sample_indices = list(range(self.total_samples_processed, self.total_samples_processed + batch_size))
        self.tracking_data['sample_index'].extend(sample_indices)
        self.total_samples_processed += batch_size
    
    def plot_final_training_metrics(self, save_dir="./final_plots"):
        """Plot all tracked metrics at the end of training"""
        import matplotlib.pyplot as plt
        import os
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Create 7 plots in a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        axes = axes.flatten()  # Flatten to make indexing easier
            
        sample_indices = self.tracking_data['sample_index']
        
        # Plot 1: Chosen LogPs
        axes[0].plot(sample_indices, self.tracking_data['chosen_logps'], 
                     color='blue', alpha=0.7, linewidth=1, label='Chosen LogPs')
        axes[0].set_title('Chosen Log Probabilities Across Training Samples', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Sample Index')
        axes[0].set_ylabel('Log Probability')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot 2: Chosen Entropy
        axes[1].plot(sample_indices, self.tracking_data['chosen_entropy'], 
                     color='green', alpha=0.7, linewidth=1, label='Chosen Entropy')
        axes[1].set_title('Chosen Response Entropy Across Training Samples', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Entropy')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
        # Plot 3: All Rejected LogPs
        colors = ['red', 'orange', 'purple']
        for i, key in enumerate(['rejected1', 'rejected2', 'rejected3']):
            if key in self.tracking_data['rejected_logps'] and self.tracking_data['rejected_logps'][key]:
                axes[2].plot(sample_indices, self.tracking_data['rejected_logps'][key], 
                        color=colors[i], alpha=0.7, linewidth=1, label=f'{key} LogPs')
    
        axes[2].set_title('Rejected Log Probabilities Across Training Samples', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('Sample Index')
        axes[2].set_ylabel('Log Probability')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot 4: All Rejected Entropies  
        for i, key in enumerate(['rejected1', 'rejected2', 'rejected3']):
            if key in self.tracking_data['rejected_entropy'] and self.tracking_data['rejected_entropy'][key]:
                axes[3].plot(sample_indices, self.tracking_data['rejected_entropy'][key], 
                            color=colors[i], alpha=0.7, linewidth=1, label=f'{key} Entropy')
        
        axes[3].set_title('Rejected Response Entropies Across Training Samples', fontsize=12, fontweight='bold')
        axes[3].set_xlabel('Sample Index')
        axes[3].set_ylabel('Entropy')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        # Plot 5: Chosen - Rejected Logratios Differences
        for i, key in enumerate(['rejected1', 'rejected2', 'rejected3']):
            if key in self.tracking_data['rejected_logratios'] and self.tracking_data['rejected_logratios'][key]:
                diff_values = [c - r for c, r in zip(self.tracking_data['chosen_logratios'], 
                                                    self.tracking_data['rejected_logratios'][key])]
                axes[4].plot(sample_indices, diff_values, 
                            color=colors[i], alpha=0.7, linewidth=1, label=f'Chosen - {key}')
        
        axes[4].set_title('Chosen - Rejected Logratios Differences', fontsize=12, fontweight='bold')
        axes[4].set_xlabel('Sample Index')
        axes[4].set_ylabel('Logratios Difference')
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()
        
        # Plot 6: Rejected - Chosen Entropy Differences
        for i, key in enumerate(['rejected1', 'rejected2', 'rejected3']):
            if key in self.tracking_data['rejected_entropy'] and self.tracking_data['rejected_entropy'][key]:
                diff_values = [r - c for r, c in zip(self.tracking_data['rejected_entropy'][key], 
                                                    self.tracking_data['chosen_entropy'])]
                axes[5].plot(sample_indices, diff_values, 
                            color=colors[i], alpha=0.7, linewidth=1, label=f'{key} - Chosen')
        
        axes[5].set_title('Rejected - Chosen Entropy Differences', fontsize=12, fontweight='bold')
        axes[5].set_xlabel('Sample Index')
        axes[5].set_ylabel('Entropy Difference')
        axes[5].grid(True, alpha=0.3)
        axes[5].legend()
        
        # Plot 7: Temp1 and Entropy Diff Values
        axes[6].plot(range(len(self.tracking_data['temp1_values'])), self.tracking_data['temp1_values'], 
                    color='darkblue', alpha=0.7, linewidth=1, label='Temp1 Values')
        
        # Create secondary y-axis for entropy_diff
        ax6_twin = axes[6].twinx()
        ax6_twin.plot(range(len(self.tracking_data['entropy_diff_values'])), self.tracking_data['entropy_diff_values'], 
                     color='darkgreen', alpha=0.7, linewidth=1, label='Entropy Diff Values')
        
        axes[6].set_title('Temp1 and Entropy Difference Values', fontsize=12, fontweight='bold')
        axes[6].set_xlabel('Computation Index')
        axes[6].set_ylabel('Temp1 Values', color='darkblue')
        ax6_twin.set_ylabel('Entropy Diff Values', color='darkgreen')
        axes[6].grid(True, alpha=0.3)
            
        # Combine legends
        lines1, labels1 = axes[6].get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        axes[6].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
        # Hide unused subplots
        for i in range(7, 9):
            axes[i].set_visible(False)
        
        # Adjust layout and save
        plt.tight_layout(pad=3.0)
        plot_path = os.path.join(save_dir, 'final_training_metrics.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Final training metrics plot saved to {plot_path}")
        print(f"Total samples tracked: {len(sample_indices)}")
    
    def save_tracking_data_json(self, save_path="./final_tracking_data.json"):
        """Save all tracking data to JSON"""
        import json
        
        with open(save_path, 'w') as f:
            json.dump(self.tracking_data, f, indent=2)
        
        print(f"Tracking data saved to {save_path}")
    def get_batch_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        
        
        
        metrics = {}
        
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
        ) = self.concatenated_forward(model, batch)
        
    
        
        with torch.no_grad():
            (
                reference_chosen_logps,
                reference_rejected_logps,
                _,
                _,
            ) = self.concatenated_forward(self.ref_model, batch)


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

        return losses.mean(), metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        
        
        
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )
        
        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        

        if return_outputs:
            return (loss, metrics)
        return loss
    

    def generate_samples_for_evaluation(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the model and reference model for the given batch of inputs."""

       

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


        return policy_output_decoded, reference_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        
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

        

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="eval")

        

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            # "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])


        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)
        


    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """

        
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        

        
        # Add averaged stored metrics to logs
        original_logs_keys = set(logs.keys())
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        

        
        del self._stored_metrics[train_eval]
        

        
        super().log(logs, *args, **kwargs)
        
