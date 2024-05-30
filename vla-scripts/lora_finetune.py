"""
lora_finetune.py

Simple script that demonstrates parameter-efficient finetuning via HF models.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import timm
import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

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
    lora_rank: int = 8                                              # Rank of LoRA weight matrix
    lora_dropout: float = 0.1                                       # Dropout applied to LoRA weights
    use_dora: bool = False                                          # Whether to use weight-decomposed LoRA (DoRA)
    use_qlora: bool = False                                         # Whether to load model quantized for LoRA finetuning
    batch_size: int = 16                                            # Finetuning batch size
    max_steps: int = 20000                                          # Max number of finetuning steps
    learning_rate: float = 2e-5                                     # Finetuning learning rate
    grad_accumulation_steps: int = 1                                # Steps of gradient accumulation

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # Initialize
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}_{cfg.dataset_name}"
        f"_bs{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"_lr{cfg.learning_rate}_lora_r{cfg.lora_rank}"
        f"_dropout{cfg.lora_dropout}_dora{cfg.use_dora}"
    )
    run_dir = cfg.run_root_dir / exp_id
    temp_dir = Path("/tmp") / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Load pre-trained VLA
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    if cfg.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    else:
        bnb_config = None

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
    )
    if not cfg.use_qlora:
        vla = vla.to(device)

    # Wrap model in PEFT for LoRA finetuning
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
        use_dora=cfg.use_dora,
    )

    if cfg.use_qlora:
        vla = prepare_model_for_kbit_training(vla)
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    # Create optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

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

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )

    # Initialize logging
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"FT-LORA-{exp_id}-{DATE_TIME}",
    )

    with tqdm.tqdm(
        total=cfg.max_steps,
        leave=False,
    ) as progress:
        vla.train()
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"],
                )
            loss = output.loss
            loss.backward()

            # Compute accuracy and L1 loss for logging
            action_preds = output.logits[:, vla.vision_backbone.patch_embed.num_patches : -1].argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # Compute Accuracy
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # Compute L1 Loss on Predicted (Continuous) Actions
            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            action_l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)

            if step % 10 == 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "action_accuracy": action_accuracy,
                        "l1_loss": action_l1_loss,
                    },
                    step=step,
                )

            # Optimizer Step
            if (step + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                progress.update()

            if step > 0 and step % 1000 == 0:
                print(f"Saving model checkpoint, step {step}...")
                # First save LoRA adapter weights in temp directory
                vla.save_pretrained(temp_dir)

                # Merge LoRA weights into model backbone for faster inference
                base_vla = AutoModelForVision2Seq.from_pretrained(
                    cfg.vla_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
                merged_vla = PeftModel.from_pretrained(base_vla, temp_dir)
                merged_vla = merged_vla.merge_and_unload()
                merged_vla.save_pretrained(run_dir)

                # Save processor
                processor.save_pretrained(run_dir)


if __name__ == "__main__":
    finetune()
