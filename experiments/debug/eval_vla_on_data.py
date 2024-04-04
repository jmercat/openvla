"""
eval_vla_on_data.py

Runs a VLA checkpoint on samples from a dataset.
(Sanity check to ensure we are doing test-time inference correctly.)

Usage:
    python experiments/debug/eval_vla_on_data.py \
        --vla.type <VLA_TRAINING_CONFIG_NAME> \
        --data_root_dir <BASE_DATASETS_DIR> \
        --pretrained_checkpoint <CHECKPOINT_PATH>
"""

import copy
import json
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
from transformers import LlamaTokenizerFast
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, VLAConfig, VLARegistry
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.vlms import OpenVLA, PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class ModifiedOpenVLA(OpenVLA):
    """
    Subclass of OpenVLA that overrides `predict_action()`.
    """

    def __init__(
        self,
        *args,
        action_norm_stats,
        action_tokenizer,
        **kwargs,
    ) -> None:
        super(ModifiedOpenVLA, self).__init__(
            *args, action_norm_stats=action_norm_stats, action_tokenizer=action_tokenizer, **kwargs
        )

    @torch.inference_mode()
    def predict_action(self, image: Image, instruction: str, **kwargs: str) -> np.ndarray:
        """Same as `OpenVLA.predict_action()`, except that we also return `predicted_action_token_ids`."""
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)

        if isinstance(tokenizer, LlamaTokenizerFast):
            # Note (Moo Jin): We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
            # in order for the predictions to match the training configuration and be accurate.
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.device)), dim=1
            )
        else:
            # TODO (Moo Jin): figure out how to make this tokenizer-independent
            raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,  # Shape: [1, seq]
                pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                max_new_tokens=self.action_dim,
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[:, -self.action_dim :]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())

        # Unnormalize actions
        mask = self.action_norm_stats.get("mask", np.ones_like(self.action_norm_stats["mean"], dtype=bool))
        action_high, action_low = np.array(self.action_norm_stats["q99"]), np.array(self.action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions, predicted_action_token_ids


def load_vla(
    model_path,
    dataset_name,
    hf_token=None,
    cache_dir=None,
    load_for_training=False,
) -> OpenVLA:
    """Same as prismatic.models.load.load_vla() except that we load ModifiedOpenVLA instead of OpenVLA."""
    overwatch.info(f"Loading from local checkpoint path `{model_path}`")

    # Assert that the checkpoint path looks like: `..../<RUN_ID>/checkpoints/<CHECKPOINT_DIR>`
    assert os.path.isfile(model_path)
    assert model_path[-3:] == ".pt" and model_path.split("/")[-2] == "checkpoints" and len(model_path.split("/")) >= 3
    run_dir = Path("/".join(model_path.split("/")[:-2]))  # `..../<RUN_ID>`

    # Get paths for `config.json`, 'dataset_statistics.json' and pretrained checkpoint
    config_json = run_dir / "config.json"
    dataset_stats_json = run_dir / "dataset_statistics.json"
    checkpoint_pt = model_path
    assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
    assert dataset_stats_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

    # Load VLA Config from `config.json` and extract Model Config
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        model_cfg = ModelConfig.get_choice_class(vla_cfg["base_vlm"])()

    # Load dataset statistics for action de-normalization
    with open(dataset_stats_json, "r") as f:
        action_norm_stats = json.load(f)[dataset_name]["action"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg.model_id}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg.vision_backbone_id}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg.llm_backbone_id}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg.arch_specifier}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg.vision_backbone_id,
        model_cfg.image_resize_strategy,
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg.llm_backbone_id,
        llm_max_length=model_cfg.llm_max_length,
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Create action tokenizer
    action_tokenizer = ActionTokenizer(llm_backbone.get_tokenizer())

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg.model_id}[/] from Checkpoint")
    vla = ModifiedOpenVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        action_norm_stats=action_norm_stats,
        action_tokenizer=action_tokenizer,
    )

    return vla


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
    dataset_name = "bridge_orig"  # TODO (Moo Jin): CHANGE ME BASED ON VLA CONFIG
    input(f"WARNING: Using dataset statistics for dataset '{dataset_name}'. Press Enter to proceed...")
    vla = load_vla(cfg.pretrained_checkpoint, dataset_name=dataset_name, hf_token=hf_token, load_for_training=False)
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
    # Call HF-style `generate()` (via superclass of PrismaticVLM) to generate action tokens.
    generated_ids = super(PrismaticVLM, vla).generate(**inputs, max_new_tokens=ACTION_DIM, do_sample=False)
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
    Leverages method `ModifiedOpenVLA.predict_action()`.
    We use greedy decoding here by passing `do_sample=False` to `generate()`.

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
    # Extract task description.
    USER_PROMPT_START_IDX = 34
    assert (
        tokenizer.decode(inputs["input_ids"][0][:USER_PROMPT_START_IDX])
        == "<s> A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    )
    assert tokenizer.decode(inputs["input_ids"][0][: -ACTION_DIM - 2])[-10:] == "ASSISTANT:"
    NUM_TOKENS_FOR_ASSISTANT = 7
    message = tokenizer.decode(inputs["input_ids"][0][USER_PROMPT_START_IDX : -ACTION_DIM - NUM_TOKENS_FOR_ASSISTANT])
    TASK_DESCRIPTION_START_IDX = 37
    assert message[:TASK_DESCRIPTION_START_IDX] == "What action should the robot take to "
    assert message[-1] == "?"
    task_description = message[TASK_DESCRIPTION_START_IDX:-1]
    # Call `ModifiedOpenVLA.predict_action()` to generate action tokens.
    action, predicted_action_token_ids = vla.predict_action(image, task_description, do_sample=False)
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
        # Generate actions without teacher forcing: via HF-style `generate()`.
        actions_accuracy, l1_loss = eval_no_teacher_forcing(batch, vla, action_tokenizer)
        stats_dict["no_teacher_forcing"]["l1_loss"] += l1_loss
        stats_dict["no_teacher_forcing"]["actions_accuracy"] += actions_accuracy
        print("-----------------------------------------------------------")
        # Generate actions without teacher forcing: via manual autoregressive sequential prediction.
        actions_accuracy, l1_loss = eval_no_teacher_forcing_manual_generate(batch, vla, action_tokenizer)
        stats_dict["no_teacher_forcing_manual_generate"]["l1_loss"] += l1_loss
        stats_dict["no_teacher_forcing_manual_generate"]["actions_accuracy"] += actions_accuracy
        print("-----------------------------------------------------------")
        # Generate actions without teacher forcing: via `ModifiedOpenVLA.predict_action()`.
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
