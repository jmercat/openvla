"""
visualize.py

Post-hoc computes train/val metrics for VLA checkpoints and visualizes predictions.

Usage:
    python scripts/visualize.py \
        --vla.type <VLA_TRAINING_CONFIG_NAME> \
        --data_root_dir <BASE_DATASETS_DIR> \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import copy
import os
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader
import tqdm

import wandb
from prismatic.models import load_vla
from prismatic.conf import VLAConfig, VLARegistry
from prismatic.vla import get_vla_dataset_and_collator

ACTION_DIM = 7
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@dataclass
class VisualizeConfig:
    # fmt: off
    # Pre-trained VLA model checkpoint to load
    pretrained_checkpoint: Union[str, Path] = Path(
        "/shared/karl/models/open_vla/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/step-080000-epoch-09-loss=0.0987.pt"
    )

    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.LLAVA_REPRO_MX_BRIDGE.vla_id)
    )

    # Directory containing dataset(s) to run evaluations on
    data_root_dir: str = "/shared/karl/data"

    # Eval params
    eval_batch_size: int = 128
    eval_batches: int = 32

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # WandB setup
    wandb_project: str = "openvla"                              # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                      # Name of entity to log under

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    logging.info(f"Loading VLA from checkpoint: {cfg.pretrained_checkpoint}")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    vla = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLA parameter not in full precision: {param}"

    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def make_data_loader(cfg, vla, train):
    image_transform = vla.vision_backbone.get_image_transform()
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=image_transform,
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=10000,
        train=train,
    )
    # Create dataloader.
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.eval_batch_size,
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
    )
    return dataloader, action_tokenizer


def compute_metrics(
    action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids
):
    """
    Returns tuple (action tokens accuracy, L1 loss) given predicted and ground-truth action token IDs.
    """
    # Compute action tokens accuracy
    actions_accuracy = (
        (predicted_action_token_ids == ground_truth_action_token_ids).type(torch.bfloat16)
    ).item()

    # Compute L1 loss
    ground_truth_actions = (
        action_tokenizer.decode_token_ids_to_actions(ground_truth_action_token_ids.cpu().numpy())
    )
    predicted_actions = (
        action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(
        torch.Tensor(predicted_actions),
        torch.Tensor(ground_truth_actions),
        reduction="none"
    ).item()

    return {
        "accuracy": actions_accuracy,
        "l1_loss": l1_loss,
    }


def eval_on_batch(batch, vla, action_tokenizer):
    """
    Evaluates model on input batch without using teacher forcing.
    Leverages the model's `generate()` function.
    We use greedy decoding here by passing `do_sample=False` to `generate()`.
    """
    # Prepare inputs.
    inputs = copy.deepcopy(batch)

    # Remove the action tokens from the prompt.
    ground_truth_action_token_ids = inputs["labels"][:, -1 - ACTION_DIM : -1]
    inputs["input_ids"] = inputs["input_ids"][:, : -ACTION_DIM - 1]
    inputs["labels"] = inputs["labels"][:, : -ACTION_DIM - 1]
    inputs["attention_mask"] = inputs["attention_mask"][:, : -ACTION_DIM - 1]

    # Call `generate()` to generate action tokens.
    generated_ids = vla.generate(**inputs, max_new_tokens=ACTION_DIM, do_sample=False)
    predicted_action_token_ids = generated_ids[:, -ACTION_DIM:]

    # Compute action tokens accuracy and L1 loss.
    return compute_metrics(
        action_tokenizer,
        ground_truth_action_token_ids,
        predicted_action_token_ids,
    )


def eval_on_dataset(data_loader, vla, action_tokenizer, cfg):
    metrics = None
    with tqdm.tqdm(total=cfg.eval_batches, desc="Eval steps") as progress:
        for idx, batch in enumerate(data_loader):
            # Prepare inputs and move them to device
            batch.pop("dataset_names")
            for k in batch.keys():
                batch[k] = batch[k].to(DEVICE)
                if k == "pixel_values":
                    batch[k] = batch[k].to(dtype=vla.llm_backbone.half_precision_dtype)

            # Generate actions via autoregressive sampling: via `generate()`
            batch_metrics = eval_on_batch(batch, vla, action_tokenizer)
            if metrics is None:
                metrics = batch_metrics
            else:
                for k in metrics:
                    metrics[k] = np.concatenate((metrics[k], batch_metrics[k]))

            if idx == cfg.eval_batches - 1:
                break

            progress.update()

    # compute averages globally and per dimension
    log_metrics = {}
    for key in metrics:
        log_metrics[f"avg_{key}"] = metrics[key].mean()
        if len(metrics[key].shape) == 2:
            for dim in range(metrics[key].shape[1]):
                log_metrics[f"dim{dim + 1}_{key}"] = metrics[key].mean(axis=1)[dim]

    return log_metrics


@draccus.wrap()
def visualize_policy(cfg: VisualizeConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # Initialize WandB
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"VIS-{cfg.vla.vla_id}-{DATE_TIME}",
    )

    # Get VLA policy
    vla = get_vla(cfg)

    # Get VLA dataset and collator for train and val
    logging.info(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    train_loader, action_tokenizer = make_data_loader(cfg, vla, train=True)
    val_loader, _ = make_data_loader(cfg, vla, train=False)

    # Dataset evaluations
    logging.info("Running offline evaluations...")
    train_metrics = eval_on_dataset(train_loader, vla, action_tokenizer, cfg)
    wandb.log({"train": train_metrics})
    val_metrics = eval_on_dataset(val_loader, vla, action_tokenizer, cfg)
    wandb.log({"val": val_metrics})


if __name__ == "__main__":
    visualize_policy()


