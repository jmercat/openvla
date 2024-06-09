"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA.

With LoRA fine-tuning:
- a single 48GB GPU can fit batch size up to 12
- a single 80GB GPU can fit batch size up to 24

Usage:
    To launch training with the default config, simply run:
        torchrun --nproc-per-node <NUM_GPUS> vla-scripts/finetune.py

    You also can overwrite default config values on the command line, like so:
        torchrun --nproc-per-node <NUM_GPUS> vla-scripts/finetune.py \
            --data_root_dir /PATH/TO/RLDS/DATASETS/DIR \
            --dataset_name <DATASET_NAME> \
            --run_root_dir /PATH/TO/LOGS/DIR \
            ...
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import timm
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def create_vision_transform(vla, input_size):
    """Gets image transform for the vision encoder."""
    data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
    data_cfg["input_size"] = (3, input_size, input_size)
    return timm.data.create_transform(
        input_size=data_cfg["input_size"],
        interpolation=data_cfg["interpolation"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        crop_pct=1.0,  # Set to 1.0 to disable cropping
        crop_mode="center",  # Default crop mode --> no-op when `crop_pct == 1.0`
        is_training=False,  # Disable image aug when loading transform; image aug is handled by RLDS dataloader
    )


@dataclass
class FinetuneConfig:
    # fmt: off

    vla_path: str = "openvla/openvla-7b-v01"                        # Path to HuggingFace OpenVLA model

    # Directory Paths
    data_root_dir: Path = Path(                                     # Path to Open-X dataset directory
        "/raid/datasets"
    )
    dataset_name: str = "droid_wipe"                                # Name of dataset to fine-tune on
    run_root_dir: Path = Path("/raid/users/karl/models")            # Path to directory to store logs & checkpoints
    temp_root_dir: Path = Path("/tmp")                              # Temp dir for storing LoRA adapter before fusing

    # Fine-tuning arguments
    batch_size: int = 16                                            # Fine-tuning batch size
    max_steps: int = 200_000                                        # Max number of fine-tuning steps
    save_steps: int = 5000                                          # Interval for checkpoint saving
    learning_rate: float = 2e-5                                     # Fine-tuning learning rate
    grad_accumulation_steps: int = 1                                # Steps of gradient accumulation
    image_aug: bool = True                                          # Whether to train with image augmentations
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # LoRA arguments
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to load model quantized for LoRA fine-tuning
                                                                    # CAUTION: reduces memory, but hurts performance

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Environment variable or Path to HF Token

    # Tracking Parameters
    wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                          # Name of entity to log under


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    # Set up experiment ID and log directory
    distributed_state = PartialState()
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}_{cfg.dataset_name}"
        f"_bs{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"_lr{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"_lora_r{cfg.lora_rank}" f"_dropout{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "_4bit"
    run_dir = cfg.run_root_dir / exp_id
    temp_dir = cfg.temp_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # Load pre-trained VLA
    assert torch.cuda.is_available()
    device = distributed_state.local_process_index
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)

    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning."
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        # BnB handles device placement automatically for quantized training
        vla = vla.to(device)

    # Wrap model in PEFT for LoRA fine-tuning
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

    # Manually set requires_grad = False for unused final-layer params in vision encoder
    # to not trip up DDP (we use the second-to-last layer features)
    for module in [vla.vision_backbone.attn_pool, vla.vision_backbone.norm, vla.vision_backbone.blocks[-1]]:
        for param in module.parameters():
            param.requires_grad = False

    # Wrap VLA in PyTorch DDP wrapper for multi-GPU training
    vla = DDP(vla, device_ids=[device], gradient_as_bucket_view=True)

    # Create optimizer
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Load training data
    # We use Open X-Embodiment RLDS dataset by default
    # If you instead wish to use a simple custom PyTorch Dataset, without RLDS, see below
    # Note: Our training code does not include an outer for loop iterating through multiple epochs because
    #       RLDS datasets loop infinitely and progress to the next epoch automatically. If you opt to use a
    #       PyTorch Dataset instead of RLDS, you should add an epoch loop below.

    ################################ Example PyTorch Dataset ################################
    # from prismatic.vla.datasets import DummyDataset
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=create_vision_transform(vla.module, vla.module.config.image_size),
    #     prompt_builder_fn=VicunaV15ChatPromptBuilder,
    # )
    #########################################################################################

    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=create_vision_transform(vla.module, vla.module.config.image_size),
        prompt_builder_fn=VicunaV15ChatPromptBuilder,
    )
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=(vla.module.config.image_size, vla.module.config.image_size),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # Save dataset statistics for inference action de-normalization
    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    # Create collator and training data loader
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # important: set to 0 since RLDS is handling parallelism
    )

    # Initialize logging
    if distributed_state.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"FT-{exp_id}-{DATE_TIME}",
        )

    # Begin training
    with tqdm.tqdm(
        total=cfg.max_steps,
        leave=False,
    ) as progress:
        vla.train()
        optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            # Compute forward pass in half precision and get train loss
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
            action_preds = output.logits[:, vla.module.vision_backbone.patch_embed.num_patches : -1].argmax(dim=2)
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

            # Log metrics on wandb
            if distributed_state.is_main_process and step % 10 == 0:
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

            # Save model checkpoint
            # By default, keeps only the latest checkpoint and continually overrides it
            if step > 0 and step % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving model checkpoint, step {step}...")
                    # If using LoRA we first save adapter weights, then merge weights for faster inference
                    # Otherwise, we simply save the weights
                    save_dir = temp_dir if cfg.use_lora else run_dir
                    vla.module.save_pretrained(save_dir)

                    # Merge LoRA weights into model backbone for faster inference
                    if cfg.use_lora:
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
                dist.barrier()


if __name__ == "__main__":
    finetune()
