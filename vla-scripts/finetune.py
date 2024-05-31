"""
finetune.py

Simple script that demonstrates finetuning a pre-trained OpenVLA model via the HF Trainer.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import timm
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments

import wandb
from prismatic.models.backbones.llm.prompting import VicunaV15ChatPromptBuilder
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def create_vision_transform(vla, input_size):
    data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
    data_cfg["input_size"] = (3, input_size, input_size)
    return timm.data.create_transform(
        input_size=data_cfg["input_size"],
        interpolation=data_cfg["interpolation"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        crop_pct=1.0,  # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
        crop_mode="center",  # Default crop mode -- no-op when `crop_pct == 1.0`
        is_training=False,  # No image augmentations when loading the transform!
    )


class TrainLogTrainer(Trainer):
    """Subclass Huggingface Trainer to log values on train set."""

    def __init__(self, *args, **kwargs):
        self._train_log_function = kwargs.pop("train_log_function")
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        self.log(self._train_log_function(inputs, outputs))
        return (loss, outputs) if return_outputs else loss


@dataclass
class FinetuneConfig:
    # fmt: off

    vla_path: str = "openvla/openvla-7b-v01"                        # Path to HuggingFace OpenVLA model

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "/raid/datasets"
    )
    dataset_name: str = "droid_wipe"                                # Name of dataset to finetune on
    run_root_dir: Path = Path("/raid/users/karl/models")            # Path to directory to store logs & checkpoints

    # Finetune arguments
    per_device_batch_size: int = 16                                 # Finetuning batch size per device
    max_steps: int = 30000                                          # Max number of finetuning steps
    learning_rate: float = 2e-5                                     # Finetuning learning rate
    grad_accumulation_steps: int = 1                                # Steps of gradient accumulation

    # LoRA parameters
    use_lora: bool = False                                          # Whether to use LoRA during finetuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    load_in_8bit: bool = False                                      # Load model in 8 bit for QLoRA finetuning
    load_in_4bit: bool = False                                      # Load model in 4 bit for QLoRA finetuning

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    if cfg.load_in_8bit or cfg.load_in_4bit:
        assert cfg.use_lora, "Only support 8bit or 4bit finetuning when using LoRA"

    # Initialize
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}_{cfg.dataset_name}"
        f"_per_device_bs{cfg.per_device_batch_size * cfg.grad_accumulation_steps}"
        f"_lr{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"_lora_r{cfg.lora_rank}_dropout{cfg.lora_dropout}" f"_8bit_{cfg.load_in_8bit}_4bit_{cfg.load_in_4bit}"
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Load pre-trained VLA
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if not cfg.load_in_8bit or cfg.load_in_4bit:
        vla = vla.to(device)

    # Wrap model in PEFT for LoRA finetuning
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # Load training data
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.dataset_name,
        image_transform=create_vision_transform(vla, vla.config.image_size),
        tokenizer=processor.tokenizer,
        prompt_builder_fn=VicunaV15ChatPromptBuilder,
        default_image_resolution=(3, vla.config.image_size, vla.config.image_size),
        shuffle_buffer_size=100_000,
        image_aug=True,
    )
    save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Overwrite dataset statistics in model with stats of finetuning dataset
    # (these statistics will be used to un-normalize action predictions during inference)
    vla.norm_stats = vla_dataset.dataset_statistics

    # Define function for logging train accuracy and L1 loss
    def compute_train_metrics(inputs, outputs):
        action_preds = outputs.logits[:, vla.vision_backbone.patch_embed.num_patches : -1].argmax(dim=2)
        action_gt = inputs["labels"][:, 1:].to(action_preds.device)
        mask = action_gt > action_tokenizer.action_token_begin_idx

        # Compute Accuracy
        correct_preds = (action_preds == action_gt) & mask
        action_accuracy = correct_preds.sum().float() / mask.sum().float()

        # Compute L1 Loss on Predicted (Continuous) Actions
        continuous_actions_pred = torch.tensor(
            action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
        )
        continuous_actions_gt = torch.tensor(action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy()))
        action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
        return {
            "action_accuracy": action_accuracy,
            "l1_loss": action_l1_loss,
        }

    # Initialize logging
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"FT-OPENVLA-{exp_id}-{DATE_TIME}",
    )

    # Create HuggingFace Trainer
    args = TrainingArguments(
        output_dir=run_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        seed=42,
        report_to="wandb",
        bf16=True,
        learning_rate=cfg.learning_rate,
        gradient_accumulation_steps=cfg.grad_accumulation_steps,
        gradient_checkpointing=True,
        max_steps=cfg.max_steps,
        logging_strategy="steps",
        logging_steps=1,
        save_steps=1000,
        save_total_limit=1,  # only keep most recent checkpoint
    )
    trainer = TrainLogTrainer(
        model=vla,
        args=args,
        train_dataset=vla_dataset,
        data_collator=collator,
        train_log_function=compute_train_metrics,
        tokenizer=processor,
    )

    # Run training
    trainer.train()


if __name__ == "__main__":
    finetune()
