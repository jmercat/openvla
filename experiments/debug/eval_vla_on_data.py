"""
eval_vla_on_bridge_data.py

Runs a VLA checkpoint on samples from a dataset.
(Sanity check to ensure we are doing test-time inference correctly.)

Usage:
    python experiments/debug/eval_vla_on_data.py \
        --vla.type <VLA_TRAINING_CONFIG_NAME> \
        --data_root_dir <BASE_DATASETS_DIR> \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import draccus
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.conf import DatasetConfig, DatasetRegistry, VLAConfig, VLARegistry
from prismatic.models import load_vla
from prismatic.vla import get_vla_dataset_and_collator

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


@dataclass
class GenerateConfig:
    # fmt: off

    # VLAConfig (`prismatic/conf/vla.py`); override with --vla.type `VLARegistry.<VLA>.vla_id`
    vla: VLAConfig = field(
        default_factory=VLAConfig.get_choice_class(VLARegistry.LLAVA_REPRO_MX_BRIDGE.vla_id)
    )

    # Directory containing dataset(s) to run evaluations on
    data_root_dir: str = "/scr-ssd/moojink/data/oxe/modified/"


    # Pre-trained VLA model checkpoint to load
    pretrained_checkpoint: Union[str, Path] = Path(
        "/sphinx/u/moojink/prismatic-vlms/logs/bridge--repro-llava-batching-wd-p1+7b--stage=vla_finetune--seed=7--2024_01_20/checkpoints/step-065000-epoch-00-loss=0.4670.pt"
    )

    # DatasetConfig from `prisma/conf/datasets.py`; override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id))

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Prepare for model loading.
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    # Load VLA checkpoint.
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vla = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
    for param in vla.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
    # Cast to half precision.
    vla.vision_backbone.to(dtype=vla.vision_backbone.half_precision_dtype)
    vla.llm_backbone.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(dtype=vla.llm_backbone.half_precision_dtype)
    vla.to(DEVICE)
    return vla


def compute_actions_accuracy_l1_loss(
    action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids, print_text
):
    """
    Returns tuple (action tokens accuracy, L1 loss) given predicted and ground-truth action token IDs.
    """
    # Compute action tokens accuracy.
    actions_accuracy = ((predicted_action_token_ids == ground_truth_action_token_ids).type(torch.bfloat16).mean()).item()
    # Compute L1 loss.
    ground_truth_actions = action_tokenizer.decode_token_ids_to_actions(ground_truth_action_token_ids.cpu().numpy())
    predicted_actions = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())
    l1_loss = torch.nn.functional.l1_loss(torch.Tensor(predicted_actions), torch.Tensor(ground_truth_actions)).item()
    print(print_text)
    print(f"    actions_accuracy: {actions_accuracy:.3f}")
    print(f"    l1_loss: {l1_loss:.3f}")
    return actions_accuracy, l1_loss


def eval_teacher_forcing(batch, vla, action_tokenizer):
    """
    Evaluates model on input batch with teacher forcing (ground-truth output token fed as inputs during generation).
    We use greedy decoding here via `argmax()` on the output logits.
    This should return similar metrics as seen during training.
    """
    inputs = copy.deepcopy(batch)
    # Run model forward pass.
    output: CausalLMOutputWithPast = vla(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        labels=inputs["labels"],
    )
    # Note (Moo Jin): Output predictions and labels are shifted by 1 position w.r.t. each other,
    # hence the different indexing below.
    ground_truth_action_token_ids = inputs["labels"][:, -1 - ACTION_DIM : -1]
    predicted_action_token_ids = output["logits"].argmax(axis=2)[:, -2 - ACTION_DIM : -2]
    # Compute action tokens accuracy and L1 loss.
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(
        action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids, print_text="Teacher forcing:"
    )
    return actions_accuracy, l1_loss


def eval_no_teacher_forcing(batch, vla, action_tokenizer):
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
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(
        action_tokenizer,
        ground_truth_action_token_ids,
        predicted_action_token_ids,
        print_text="No teacher forcing - generate():",
    )
    return actions_accuracy, l1_loss


def eval_no_teacher_forcing_manual_generate(batch, vla, action_tokenizer):
    """
    Evaluates model on input batch without using teacher forcing.
    Manually rolls out autoregressive output prediction using model's `forward()` instead of `generate()`.
    We use greedy decoding here via `argmax()` on the output logits at each step.
    """
    # Prepare inputs.
    inputs = copy.deepcopy(batch)
    # Remove the action tokens from the prompt.
    ground_truth_action_token_ids = inputs["labels"][:, -1 - ACTION_DIM : -1]
    inputs["input_ids"] = inputs["input_ids"][:, : -ACTION_DIM - 1]
    inputs["labels"] = inputs["labels"][:, : -ACTION_DIM - 1]
    inputs["attention_mask"] = inputs["attention_mask"][:, : -ACTION_DIM - 1]
    # Manually generate the outputs one token at a time, appending each
    # output token to the inputs (slow because not caching).
    curr_input_ids = torch.clone(inputs["input_ids"]).cpu().numpy().tolist()[0]
    curr_attention_mask = torch.clone(inputs["attention_mask"]).cpu().numpy().tolist()[0]
    for _ in range(ACTION_DIM):
        curr_input_ids_tensor = (
            torch.Tensor([curr_input_ids]).to(inputs["input_ids"].dtype).to(inputs["input_ids"].device)
        )
        curr_attention_mask_tensor = (
            torch.Tensor([curr_attention_mask]).to(inputs["attention_mask"].dtype).to(inputs["attention_mask"].device)
        )
        output = vla(
            input_ids=curr_input_ids_tensor,
            attention_mask=curr_attention_mask_tensor,
            pixel_values=inputs["pixel_values"],
            labels=None,
        )
        predicted_token = output["logits"].argmax(axis=2)[:, -1].item()  # greedy decoding
        curr_attention_mask.append(True)
        curr_input_ids.append(predicted_token)  # autoregressive: insert our last predicted token
    generated_ids = torch.Tensor([curr_input_ids]).to(inputs["input_ids"].dtype).to(inputs["input_ids"].device)
    predicted_action_token_ids = generated_ids[:, -ACTION_DIM:]
    # Compute action tokens accuracy and L1 loss.
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(
        action_tokenizer,
        ground_truth_action_token_ids,
        predicted_action_token_ids,
        print_text="No teacher forcing - manual autoregressive generation:",
    )
    return actions_accuracy, l1_loss


def eval_no_teacher_forcing_prompt_builder(batch, vla, action_tokenizer, tokenizer, image_transform):
    """
    Evaluates model on input batch without using teacher forcing.
    Leverages Prismatic prompt builder and `generate_with_prompt()`.
    We use greedy decoding here by passing `do_sample=False` to `generate_with_prompt()`.

    Pretty hacky because we need to recover the original image from `pixel_values` somehow (via un-normalization).
    """
    inputs = copy.deepcopy(batch)
    assert inputs["input_ids"].shape[0] == 1, "Prompt builder generate expects batch size == 1"
    # Get the original image normalization function to invert.
    orig_norm = image_transform.transforms[-1]
    assert isinstance(orig_norm, Normalize)
    # Get the un-normalization function.
    unnormalize = Normalize((-orig_norm.mean / orig_norm.std).tolist(), (1.0 / orig_norm.std).tolist())
    # Un-normalize image.
    assert inputs["pixel_values"].shape[0] == 1
    image = unnormalize(inputs["pixel_values"][0])
    image = np.uint8(
        np.transpose(unnormalize(inputs["pixel_values"][0]).type(torch.float32).cpu().numpy(), (1, 2, 0)) * 255
    )
    image = Image.fromarray(image).convert("RGB")
    # Build the input prompt.
    prompt_builder = vla.get_prompt_builder()
    TASK_DESCRIPTION_START_IDX = 34
    assert (
        tokenizer.decode(inputs["input_ids"][0][:TASK_DESCRIPTION_START_IDX])
        == "<s> A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    )
    assert tokenizer.decode(inputs["input_ids"][0][: -ACTION_DIM - 2])[-10:] == "ASSISTANT:"
    NUM_TOKENS_FOR_ASSISTANT = 7
    message = tokenizer.decode(
        inputs["input_ids"][0][TASK_DESCRIPTION_START_IDX : -ACTION_DIM - NUM_TOKENS_FOR_ASSISTANT]
    )
    prompt_builder.add_turn(role="human", message=message)
    prompt_text = prompt_builder.get_prompt()
    # Call `generate_with_prompt()` to generate action tokens.
    generated_text = vla.generate_with_prompt(image, prompt_text, max_new_tokens=ACTION_DIM, do_sample=False)
    predicted_action_token_ids = torch.unsqueeze(
        torch.Tensor(tokenizer(generated_text)["input_ids"][-ACTION_DIM:]).long(), dim=0
    ).to(DEVICE)
    # Compute action tokens accuracy and L1 loss.
    ground_truth_action_token_ids = inputs["labels"][:, -1 - ACTION_DIM : -1]
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(
        action_tokenizer,
        ground_truth_action_token_ids,
        predicted_action_token_ids,
        print_text="No teacher forcing - prompt builder:",
    )
    return actions_accuracy, l1_loss


def print_results(stats_dict, num_batches):
    """Computes aggregate loss and accuracy and prints the results."""
    for k in stats_dict.keys():
        stats_dict[k]["avg_l1_loss"] = stats_dict[k]["l1_loss"] / num_batches
        stats_dict[k]["avg_actions_accuracy"] = stats_dict[k]["actions_accuracy"] / num_batches
    print("===========================================================")
    print("Aggregate metrics:")
    print("===========================================================")
    print(f"# batches: {num_batches}")
    for k in stats_dict.keys():
        if k == "teacher_forcing":
            print("Teacher forcing:")
        elif k == "no_teacher_forcing":
            print("No teacher forcing - generate():")
        elif k == "no_teacher_forcing_manual_generate":
            print("No teacher forcing - manual autoregressive generation:")
        else:
            print("No teacher forcing - prompt builder:")
        print(f"    Average L1 loss: {stats_dict[k]['avg_l1_loss']}")
        print(f"    Average action tokens accuracy: {stats_dict[k]['avg_actions_accuracy']}")


@draccus.wrap()
def main(cfg: GenerateConfig) -> None:
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    # Get VLA policy and tokenizer.
    vla = get_vla(cfg)
    tokenizer = vla.llm_backbone.get_tokenizer()
    # Get VLA dataset and collator.
    print(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    image_transform = vla.vision_backbone.get_image_transform()
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=image_transform,
        tokenizer=vla.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vla.llm_backbone.prompt_builder_fn,
        default_image_resolution=vla.vision_backbone.default_image_resolution,
        shuffle_buffer_size=1000,
    )
    # Create dataloader.
    batch_size = 1
    dataloader = DataLoader(
        vla_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
    )
    # Prepare aggregate metrics dict.
    stats_dict = {
        "teacher_forcing": {},
        "no_teacher_forcing": {},
        "no_teacher_forcing_manual_generate": {},
        "teacher_forcing_prompt_builder": {},
    }
    for k in stats_dict.keys():
        stats_dict[k]["l1_loss"] = 0.0
        stats_dict[k]["actions_accuracy"] = 0.0
        stats_dict[k]["avg_l1_loss"] = 0.0
        stats_dict[k]["avg_actions_accuracy"] = 0.0
    num_batches = 0
    for idx, batch in enumerate(dataloader):
        print("===========================================================")
        print(f"Batch {idx}:")
        print("===========================================================")
        # Prepare inputs and move them to device.
        batch.pop("dataset_names")
        for k in batch.keys():
            batch[k] = batch[k].to(DEVICE)
            if k == "pixel_values":
                batch[k] = batch[k].to(dtype=vla.llm_backbone.half_precision_dtype)
        # Generate actions with teacher forcing.
        actions_accuracy, l1_loss = eval_teacher_forcing(batch, vla, action_tokenizer)
        stats_dict["teacher_forcing"]["l1_loss"] += l1_loss
        stats_dict["teacher_forcing"]["actions_accuracy"] += actions_accuracy
        print("-----------------------------------------------------------")
        # Generate actions without teacher forcing: via `generate()`.
        actions_accuracy, l1_loss = eval_no_teacher_forcing(batch, vla, action_tokenizer)
        stats_dict["no_teacher_forcing"]["l1_loss"] += l1_loss
        stats_dict["no_teacher_forcing"]["actions_accuracy"] += actions_accuracy
        print("-----------------------------------------------------------")
        # Generate actions without teacher forcing: via manual autoregressive sequential prediction.
        actions_accuracy, l1_loss = eval_no_teacher_forcing_manual_generate(batch, vla, action_tokenizer)
        stats_dict["no_teacher_forcing_manual_generate"]["l1_loss"] += l1_loss
        stats_dict["no_teacher_forcing_manual_generate"]["actions_accuracy"] += actions_accuracy
        print("-----------------------------------------------------------")
        # Generate actions without teacher forcing: via prompt builder.
        if batch_size == 1:
            actions_accuracy, l1_loss = eval_no_teacher_forcing_prompt_builder(
                batch, vla, action_tokenizer, tokenizer, image_transform
            )
            stats_dict["teacher_forcing_prompt_builder"]["l1_loss"] += l1_loss
            stats_dict["teacher_forcing_prompt_builder"]["actions_accuracy"] += actions_accuracy
        num_batches += 1
        if num_batches == 100:
            break
    # Compute aggregrate metrics and print the results.
    print_results(stats_dict, num_batches)


if __name__ == "__main__":
    main()
