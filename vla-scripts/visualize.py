"""
visualize.py

Post-hoc computes train/val metrics for VLA checkpoints and visualizes predictions.

Usage:
    python vla-scripts/visualize.py \
        --vla.type <VLA_TRAINING_CONFIG_NAME> \
        --data_root_dir <BASE_DATASETS_DIR> \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import matplotlib

matplotlib.use("Agg")

import copy
import glob
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import draccus
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import wandb
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.utils.data import DataLoader

from prismatic.conf import VLAConfig, VLARegistry
from prismatic.models import load_vla
from prismatic.models.vlms import OpenVLA, PrismaticVLM
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES

# === Setup ===
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


@dataclass
class VisualizeConfig:
    # fmt: off
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.LLAVA_REPRO_MX_BRIDGE.vla_id)
    )

    # Model & Data Parameters
    pretrained_checkpoint: Union[str, Path] = Path(             # Pretrained VLA checkpoint to load
        "/shared/karl/models/open_vla/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/step-080000-epoch-09-loss=0.0987.pt"
    )
    data_root_dir: str = "/shared/karl/data"                    # Directory containing dataset(s) to evaluate

    # Evaluation Parameters
    eval_samples: int = 1024                                    # Number of samples to compute statistics over
    eval_episodes: int = 5                                      # Number of episodes to visualize
    max_episode_steps: int = 80                                 # Maximum number of steps to visualize per episode
    eval_datasets: List[str] = field(default_factory=list)      # Individual datasets to visualize

    # Model Parameters
    action_dim: int = 7

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Weights & Biases
    wandb_project: str = "openvla"                              # Name of W&B project to log to (use default!)
    wandb_entity: str = "stanford-voltron"                      # Name of entity to log under

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


def get_checkpoint_paths(cfg):
    if cfg.pretrained_checkpoint.endswith(".pt"):
        # Passed a single checkpoint
        return [cfg.pretrained_checkpoint]
    else:
        return sorted(glob.glob(os.path.join(cfg.pretrained_checkpoint, "*.pt")))


def get_vla(checkpoint_path, cfg):
    """Loads and returns a VLA model from checkpoint."""
    logging.info(f"Loading VLA from checkpoint: {checkpoint_path}")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    vla: OpenVLA = load_vla(checkpoint_path, hf_token=hf_token, load_for_training=False)
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLA parameter not in full precision: {param}"

    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def make_data_loader(cfg, vla, data_mix, train):
    image_transform = vla.vision_backbone.get_image_transform()
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        data_mix,
        image_transform=image_transform,
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=100000,
        train=train,
    )
    # Create dataloader.
    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,  # currently only support generate() with batch size 1
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
    )
    return dataloader, action_tokenizer


def make_episode_dataset(cfg, vla, data_mix, train):
    image_transform = vla.vision_backbone.get_image_transform()
    vla_dataset, action_tokenizer, _ = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        data_mix,
        image_transform=image_transform,
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=10000,
        train=train,
        episodic=True,
    )
    # Create dataloader.
    dataloader = DataLoader(
        vla_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    )
    return dataloader, action_tokenizer


def compute_metrics(action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids):
    """
    Returns tuple (action tokens accuracy, L1 loss) given predicted and ground-truth action token IDs.
    """
    # Compute action tokens accuracy
    actions_accuracy = (predicted_action_token_ids == ground_truth_action_token_ids).cpu().numpy()

    # Compute L1 loss
    ground_truth_actions = action_tokenizer.decode_token_ids_to_actions(ground_truth_action_token_ids.cpu().numpy())
    predicted_actions = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())
    l1_loss = (
        torch.nn.functional.l1_loss(
            torch.Tensor(predicted_actions), torch.Tensor(ground_truth_actions), reduction="none"
        )
        .cpu()
        .numpy()
    )

    return {
        "accuracy": actions_accuracy,
        "l1_loss": l1_loss,
    }


def run_on_batch(batch, vla, action_dim):
    """
    Evaluates model on input batch without using teacher forcing.
    Leverages the model's `generate()` function.
    We use greedy decoding here by passing `do_sample=False` to `generate()`.
    """
    # Prepare inputs.
    inputs = copy.deepcopy(batch)

    # Remove the action tokens from the prompt.
    ground_truth_action_token_ids = inputs["labels"][:, -1 - action_dim : -1]
    inputs["input_ids"] = inputs["input_ids"][:, : -action_dim - 1]
    inputs["labels"] = inputs["labels"][:, : -action_dim - 1]
    inputs["attention_mask"] = inputs["attention_mask"][:, : -action_dim - 1]

    # Call `super().generate()` to generate action tokens w/o teacher forcing.
    generated_ids = super(PrismaticVLM, vla).generate(**inputs, max_new_tokens=action_dim, do_sample=False)
    predicted_action_token_ids = generated_ids[:, -action_dim:]
    return ground_truth_action_token_ids, predicted_action_token_ids


def eval_on_dataset(data_loader, vla, action_tokenizer, cfg):
    metrics = None
    with tqdm.tqdm(total=cfg.eval_samples, desc="Eval steps") as progress:
        for idx, batch in enumerate(data_loader):
            # Prepare inputs and move them to device
            batch.pop("dataset_names")
            for k in batch.keys():
                batch[k] = batch[k].to(DEVICE)
                if k == "pixel_values":
                    batch[k] = batch[k].to(dtype=vla.llm_backbone.half_precision_dtype)

            # Generate actions via autoregressive sampling: via `generate()`
            gt_actions, pred_actions = run_on_batch(batch, vla, cfg.action_dim)
            batch_metrics = compute_metrics(
                action_tokenizer,
                gt_actions,
                pred_actions,
            )
            if metrics is None:
                metrics = batch_metrics
            else:
                for k in metrics:
                    metrics[k] = np.concatenate((metrics[k], batch_metrics[k]))

            progress.update()
            if idx == cfg.eval_samples - 1:
                break

    # compute averages globally and per dimension
    log_metrics = {}
    for key in metrics:
        log_metrics[f"avg_{key}"] = metrics[key].mean()
        if len(metrics[key].shape) == 2:
            for dim in range(metrics[key].shape[1]):
                log_metrics[f"dim{dim + 1}_{key}"] = metrics[key].mean(axis=0)[dim]

    return log_metrics


class WandBFigure:
    def __init__(self, save_to=None, **figure_kwargs):
        self.fig = plt.figure(**figure_kwargs)
        self.canvas = FigureCanvas(self.fig)

    def __enter__(self):
        return plt.figure(self.fig.number)

    def __exit__(self, exc_type, exc_value, traceback):
        self.canvas.draw()
        out_image = np.frombuffer(self.canvas.tostring_rgb(), dtype="uint8")
        self.image = out_image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(self.fig)


def plot_action_episodes(gt_actions, pred_actions):
    n_act_dims = gt_actions.shape[-1]
    grid_size = int(np.ceil(np.sqrt(n_act_dims)))
    wandb_figure = WandBFigure(figsize=(grid_size * 5, grid_size * 5))
    gs = gridspec.GridSpec(grid_size, grid_size)
    with wandb_figure as fig:
        for i in range(n_act_dims):
            ax = fig.add_subplot(gs[i // grid_size, i % grid_size])
            ax.plot(gt_actions[:, i], c="b", label="action")
            ax.plot(pred_actions[:, i], c="r", label="pred")
            ax.set_ylabel(f"dim {i}")
    return wandb.Image(wandb_figure.image)


def eval_on_episodes(data_loader, vla, action_tokenizer, cfg):
    logs = {}
    with tqdm.tqdm(total=cfg.eval_episodes, desc="Eval episodes") as progress:
        for idx, episode in enumerate(data_loader):
            gt_episode_actions, pred_episode_actions = [], []
            for step in tqdm.tqdm(episode[: cfg.max_episode_steps]):
                step.pop("dataset_name")
                for k in step.keys():
                    step[k] = step[k].to(DEVICE)
                    if k == "pixel_values":
                        step[k] = step[k].to(dtype=vla.llm_backbone.half_precision_dtype)
                step["attention_mask"] = step["input_ids"].ne(vla.llm_backbone.get_tokenizer().pad_token_id)

                gt_action_tokens, pred_action_tokens = run_on_batch(step, vla, cfg.action_dim)
                gt_episode_actions.append(action_tokenizer.decode_token_ids_to_actions(gt_action_tokens.cpu().numpy()))
                pred_episode_actions.append(
                    action_tokenizer.decode_token_ids_to_actions(pred_action_tokens.cpu().numpy())
                )

            logs[f"episode_{idx}"] = plot_action_episodes(
                np.array(gt_episode_actions)[:, 0], np.array(pred_episode_actions)[:, 0]
            )
            progress.update()
            if idx == cfg.eval_episodes - 1:
                break
    return logs


def run_visualize_on_mixture(cfg, vla, data_mix, step):
    # Get VLA dataset and collator for train and val w/ full data mixture
    logging.info(f"Creating VLA Open-X Dataset with Mixture `{data_mix}`")
    train_loader, action_tokenizer = make_data_loader(cfg, vla, data_mix, train=True)
    val_loader, _ = make_data_loader(cfg, vla, data_mix, train=False)

    # Dataset evaluations
    logging.info("Running offline evaluations...")
    train_metrics = eval_on_dataset(train_loader, vla, action_tokenizer, cfg)
    wandb.log({f"train_{data_mix}/": train_metrics}, step=step)
    val_metrics = eval_on_dataset(val_loader, vla, action_tokenizer, cfg)
    wandb.log({f"val_{data_mix}/": val_metrics}, step=step)

    # Evaluate on episodic data (for individual datasets only)
    if data_mix not in OXE_NAMED_MIXTURES or len(OXE_NAMED_MIXTURES[data_mix]) == 1:
        logging.info("Running episode evaluations...")
        train_episode_loader, action_tokenizer = make_episode_dataset(cfg, vla, data_mix, train=True)
        val_episode_loader, _ = make_episode_dataset(cfg, vla, data_mix, train=False)
        train_episode_metrics = eval_on_episodes(train_episode_loader, vla, action_tokenizer, cfg)
        wandb.log({f"train_rollouts_{data_mix}/": train_episode_metrics}, step=step)
        val_episode_metrics = eval_on_episodes(val_episode_loader, vla, action_tokenizer, cfg)
        wandb.log({f"val_rollouts_{data_mix}/": val_episode_metrics}, step=step)


@draccus.wrap()
def visualize_policy(cfg: VisualizeConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # Initialize WandB
    wandb.init(
        entity=cfg.wandb_entity,
        project=cfg.wandb_project,
        name=f"VIS-{cfg.vla.vla_id}-{DATE_TIME}",
    )

    checkpoint_paths = get_checkpoint_paths(cfg)
    logging.info(f"Evaluating {len(checkpoint_paths)} checkpoint paths: {checkpoint_paths}")
    for checkpoint_path in checkpoint_paths:
        # Get checkpoint step
        step = int(os.path.basename(checkpoint_path).split("-")[1])
        logging.info(f"Evaluating checkpoint for step {step}...")

        # Get VLA policy
        vla = get_vla(checkpoint_path, cfg)

        # Run visualize script over training data mixture
        train_data_mix = cfg.vla.data_mix
        run_visualize_on_mixture(cfg, vla, train_data_mix, step)

        # Optionally run visualize script over individual datasets
        train_datasets = [m[0] for m in OXE_NAMED_MIXTURES[cfg.vla.data_mix]]
        for dataset in cfg.eval_datasets:
            if dataset not in train_datasets:
                logging.warning(
                    f"Model has not been trained with dataset {dataset}. "
                    f"Did you want to choose from {train_datasets}?"
                )
            run_visualize_on_mixture(cfg, vla, dataset, step)


if __name__ == "__main__":
    visualize_policy()
