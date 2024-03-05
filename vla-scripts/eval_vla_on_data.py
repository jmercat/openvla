"""
eval_vla_on_bridge_data.py

Runs a VLA checkpoint on samples from a dataset.
(Sanity check to ensure we are doing test-time inference correctly.)

Usage examples:
    python vla-scripts/eval_vla_on_data.py --vla.type reproduction-llava-v15+mx-bridge --data_root_dir /scr-ssd/moojink/data/oxe/modified --pretrained_checkpoint /sphinx/u/moojink/checkpoints/tri/reproduction-llava-v15+mx-bridge+n1+b32+x7/checkpoints/step-077500-epoch-00-loss=0.0488.pt
"""
import copy
import draccus
import glob
import json
import numpy as np
import os
import pickle
import requests
import torch
from accelerate.utils import set_seed
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from PIL import Image
from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry, VLAConfig, VLARegistry
from prismatic.models import load_vla
from prismatic.models.materialize import VISION_BACKBONES
from prismatic.training import VLAMetrics, get_train_strategy
from prismatic.util import set_global_seed
from prismatic.vla import get_vla_dataset_and_collator
from prismatic.vla.action_tokenizer import ActionTokenizer
# TODO clean up below
# from prisma import load_pretrained_vlm
# from prisma.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
# from prisma.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
# from prisma.models.materialize import VISION_BACKBONES
# from prisma.preprocessing import get_dataset_and_collator
# from prisma.preprocessing.action_tokenizer import ActionTokenizer
# from prisma.preprocessing.datasets.rlds.llava_rlds_dataset import (
#     rlds2llava_transform,
#     LLaVARLDSDataset
# )
# from prisma.preprocessing.datasets.rlds.oxe import make_oxe_dataset_kwargs_and_weights
# from prisma.preprocessing.datasets.rlds.oxe.oxe_dataset_mixes import OXE_NAMED_MIXES
# from prisma.util.data_utils import PaddedCollatorForLanguageModeling
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.transforms import Normalize
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm
from typing import Optional, Tuple, Type, Union


ACTION_DIM = 7
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


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

    # TODO clean up below
    # # ModelConfig from `prisma/conf/models.py`; override with --model.type `ModelRegistry.<MODEL>.model_id`
    # model: ModelConfig = field(
    #     default_factory=ModelConfig.get_choice_class(
    #         # ModelRegistry.QZ_SIGLIP_B16_256PX_NO_ALIGN_7B.model_id
    #         ModelRegistry.ONX_LLAVA_BATCHING_WD_P1_7B.model_id
    #     )
    # )


    # DatasetConfig from `prisma/conf/datasets.py`; override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id))

    # Training stage (doesn't matter here, but the loading function expects the argument)
    stage: str = "vla-finetune"

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")              # Environment variable or Path to HF Token

    # Randomness
    seed: int = 21                                              # Random Seed (for reproducibility)
    # fmt: on


# TODO clean up below
# def get_image_resize_size(vision_backbone_id: str) -> Tuple[int, int]:
#     """Gets image resize size from vision backbone ID."""
#     return VISION_BACKBONES[vision_backbone_id]["kwargs"]["default_image_size"]


def compute_actions_accuracy_l1_loss(action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids, print_text):
    """
    Returns tuple (action tokens accuracy, L1 loss) given predicted and ground-truth action token IDs.
    """
    # Compute action tokens accuracy.
    actions_accuracy = (
        (predicted_action_token_ids == ground_truth_action_token_ids)
        .type(torch.bfloat16)
        .mean()
    ).item()
    # Compute L1 loss.
    ground_truth_actions = action_tokenizer.decode_token_ids_to_actions(ground_truth_action_token_ids.cpu().numpy())
    predicted_actions = action_tokenizer.decode_token_ids_to_actions(predicted_action_token_ids.cpu().numpy())
    l1_loss = torch.nn.functional.l1_loss(torch.Tensor(predicted_actions), torch.Tensor(ground_truth_actions)).item()
    print(print_text)
    print(f"    actions_accuracy: {actions_accuracy:.3f}")
    print(f"    l1_loss: {l1_loss:.3f}")
    return actions_accuracy, l1_loss


def eval_teacher_forcing(batch, vlm, action_tokenizer, device):
    """
    Evaluates model on input batch with teacher forcing (ground-truth output token fed as inputs during generation).
    We use greedy decoding here via `argmax()` on the output logits.
    This should return similar metrics as seen during training.
    """
    # Prepare inputs.
    inputs = copy.deepcopy(batch)
    import ipdb; ipdb.set_trace()
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    # inputs["pixel_values"] = inputs["pixel_values"].to(dtype=vlm.llm_backbone.half_precision_dtype) # TODO recover
    inputs["labels"] = inputs["labels"].to(device)
    # Run model forward pass.
    output: CausalLMOutputWithPast = vlm(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        pixel_values=inputs["pixel_values"],
        labels=inputs["labels"],
    )
    # Note (Moo Jin): Output predictions and labels are shifted by 1 position w.r.t. each other,
    # hence the different indexing below.
    ground_truth_action_token_ids = inputs['labels'][:, -1-ACTION_DIM:-1]
    predicted_action_token_ids = output['logits'].argmax(axis=2)[:, -2-ACTION_DIM:-2]
    # Compute action tokens accuracy and L1 loss.
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids, print_text="Teacher forcing:")
    return actions_accuracy, l1_loss


def eval_no_teacher_forcing(batch, vlm, action_tokenizer, device):
    """
    Evaluates model on input batch without using teacher forcing.
    Leverages the model's `generate()` function.
    We use greedy decoding here by passing `do_sample=False` to `generate()`.
    """
    # Prepare inputs.
    inputs = copy.deepcopy(batch)
    inputs.pop('dataset_names')
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['labels'] = inputs['labels'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=vlm.llm_backbone.half_precision_dtype) # prevents `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the same`
    ground_truth_action_token_ids = inputs['labels'][:, -1-ACTION_DIM:-1]
    # Remove the action tokens from the prompt.
    inputs['input_ids'] = inputs['input_ids'][:, :-ACTION_DIM-1]
    inputs['labels'] = inputs['labels'][:, :-ACTION_DIM-1]
    inputs['attention_mask'] = inputs['attention_mask'][:, :-ACTION_DIM-1]
    # Call `generate()` to generate action tokens.
    generated_ids = vlm.generate(**inputs, max_new_tokens=ACTION_DIM, do_sample=False)
    predicted_action_token_ids = generated_ids[:, -ACTION_DIM:]
    # Compute action tokens accuracy and L1 loss.
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids, print_text="No teacher forcing - generate():")
    return actions_accuracy, l1_loss


def eval_no_teacher_forcing_manual_generate(batch, vlm, action_tokenizer, device):
    """
    Evaluates model on input batch without using teacher forcing.
    Manually rolls out autoregressive output prediction using model's `forward()` instead of `generate()`.
    We use greedy decoding here via `argmax()` on the output logits at each step.
    """
    # Prepare inputs.
    inputs = copy.deepcopy(batch)
    inputs.pop('dataset_names')
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['labels'] = inputs['labels'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=vlm.llm_backbone.half_precision_dtype) # prevents `RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the same`
    ground_truth_action_token_ids = inputs['labels'][:, -1-ACTION_DIM:-1]
    # Remove the action tokens from the prompt.
    inputs['input_ids'] = inputs['input_ids'][:, :-ACTION_DIM-1]
    inputs['labels'] = inputs['labels'][:, :-ACTION_DIM-1]
    inputs['attention_mask'] = inputs['attention_mask'][:, :-ACTION_DIM-1]
    # Manually generate the outputs one token at a time, appending each output token to the inputs (slow because not caching).
    curr_input_ids = torch.clone(inputs['input_ids']).cpu().numpy().tolist()[0]
    curr_attention_mask = torch.clone(inputs["attention_mask"]).cpu().numpy().tolist()[0]
    for _ in range(ACTION_DIM):
        curr_input_ids_tensor = torch.Tensor([curr_input_ids]).to(inputs['input_ids'].dtype).to(inputs['input_ids'].device)
        curr_attention_mask_tensor = torch.Tensor([curr_attention_mask]).to(inputs['attention_mask'].dtype).to(inputs['attention_mask'].device)
        output = vlm(
            input_ids=curr_input_ids_tensor,
            attention_mask=curr_attention_mask_tensor,
            pixel_values=inputs["pixel_values"],
            labels=None,
        )
        predicted_token = output['logits'].argmax(axis=2)[:, -1].item() # greedy decoding
        curr_attention_mask.append(True)
        curr_input_ids.append(predicted_token) # autoregressive: insert our last predicted token
    generated_ids = torch.Tensor([curr_input_ids]).to(inputs['input_ids'].dtype).to(inputs['input_ids'].device)
    predicted_action_token_ids = generated_ids[:, -ACTION_DIM:]
    # Compute action tokens accuracy and L1 loss.
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids, print_text="No teacher forcing - manual autoregressive generation:")
    return actions_accuracy, l1_loss


def eval_no_teacher_forcing_prompt_builder(batch, vlm, action_tokenizer, tokenizer, image_transform, device):
    """
    Evaluates model on input batch without using teacher forcing. Leverages Prismatic prompt builder and `generate_with_prompt()`.
    We use greedy decoding here by passing `do_sample=False` to `generate_with_prompt()`.

    Pretty hacky because we need to recover the original image from `pixel_values` somehow (via un-normalization).
    """
    inputs = copy.deepcopy(batch)
    # Reduce batch to 1 item (since the generate_with_prompt() function expects single sample).
    inputs["input_ids"] = inputs["input_ids"][0, :][None, ...]
    inputs["attention_mask"] = inputs["attention_mask"][0, :][None, ...]
    inputs["pixel_values"] = inputs["pixel_values"][0, ...][None, ...]
    inputs["labels"] = inputs["labels"][0, :][None, ...]
    # Prepare inputs.
    inputs["input_ids"] = inputs["input_ids"].to(device)
    inputs["attention_mask"] = inputs["attention_mask"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=vlm.llm_backbone.half_precision_dtype)
    inputs["labels"] = inputs["labels"].to(device)
    ground_truth_action_token_ids = inputs['labels'][:, -1-ACTION_DIM:-1]
    # Get the original image normalization function to invert.
    orig_norm = image_transform.transforms[-1]
    assert isinstance(orig_norm, Normalize)
    # Get the un-normalization function.
    unnormalize = Normalize((-orig_norm.mean / orig_norm.std).tolist(), (1.0 / orig_norm.std).tolist())
    # Un-normalize image.
    assert inputs["pixel_values"].shape[0] == 1
    image = unnormalize(inputs["pixel_values"][0])
    image = np.uint8(np.transpose(unnormalize(inputs["pixel_values"][0]).type(torch.float32).cpu().numpy(), (1,2,0)) * 255)
    image = Image.fromarray(image).convert("RGB")
    # Build the input prompt.
    prompt_builder = vlm.get_prompt_builder()
    TASK_DESCRIPTION_START_IDX = 34
    assert tokenizer.decode(inputs["input_ids"][0][:TASK_DESCRIPTION_START_IDX]) == "<s> A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER:"
    assert tokenizer.decode(inputs["input_ids"][0][:-ACTION_DIM-2])[-10:] == "ASSISTANT:"
    NUM_TOKENS_FOR_ASSISTANT = 7
    message = tokenizer.decode(inputs["input_ids"][0][TASK_DESCRIPTION_START_IDX:-ACTION_DIM-NUM_TOKENS_FOR_ASSISTANT])
    prompt_builder.add_turn(role="human", message=message)
    prompt_text = prompt_builder.get_prompt()
    # TODO back to generate_with_prompt???
    # Call `generate` to generate action tokens.
    generated_text = vlm.generate(image, prompt_text, max_new_tokens=ACTION_DIM, do_sample=False)
    predicted_action_token_ids = torch.unsqueeze(torch.Tensor(tokenizer(generated_text)['input_ids'][-ACTION_DIM:]).long(), dim=0).to(device)
    # Compute action tokens accuracy and L1 loss.
    actions_accuracy, l1_loss = compute_actions_accuracy_l1_loss(action_tokenizer, ground_truth_action_token_ids, predicted_action_token_ids, print_text="No teacher forcing - prompt builder:")
    return actions_accuracy, l1_loss


@draccus.wrap()
def eval(cfg: GenerateConfig) -> None:
    device = 'cuda'
    debug = False
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]

    # TODO
    # image_size = get_image_resize_size(cfg.model.vision_backbone_id)
    # resize_size = (image_size, image_size)

    # Load Base VLM from checkpoint path
    #   =>> Note :: Verifies that all parameters are loaded in FP32 on load!
    print(f"Loading VLM from checkpoint: {cfg.pretrained_checkpoint}")
    vlm = load_vla(cfg.pretrained_checkpoint, hf_token=hf_token, load_for_training=False)
    for param in vlm.parameters():
        assert param.dtype == torch.float32, f"Loaded VLM parameter not in full precision: {param}"
    # TODO: recover below
    # # Cast to half precision.
    # vlm.vision_backbone.to(dtype=vlm.vision_backbone.half_precision_dtype)
    # vlm.llm_backbone.to(dtype=vlm.llm_backbone.half_precision_dtype)
    # vlm.to(dtype=vlm.llm_backbone.half_precision_dtype)
    vlm.to(device)
    # Create tokenizer.
    tokenizer = vlm.llm_backbone.get_tokenizer()



    # TODO clean up below
    # transform = partial(
    #     rlds2llava_transform,
    #     model_family=cfg.model_family,
    #     action_tokenizer=action_tokenizer,
    #     base_tokenizer=tokenizer,
    #     image_transform=image_transform,
    #     prompt_builder_fn=llm_backbone.prompt_builder_fn,
    # )
    # # Create dataset and collator.
    # vla_data_mix = OXE_NAMED_MIXES[cfg.vla_data_mix_name]
    # data_kwargs_list, weights = make_oxe_dataset_kwargs_and_weights(
    #     data_mix=vla_data_mix,
    #     data_dir=cfg.vla_data_directory,
    #     load_camera_views=("primary",),
    #     load_depth=False,
    #     load_proprio=False,
    #     load_language=True,
    #     action_proprio_normalization_type="bounds_q99",
    # )
    # data_config = dict(
    #     traj_transform_kwargs=dict(
    #         window_size=1,                          # if we wanted to feed / predict more than one step
    #         future_action_window_size=0,        # for action chunking
    #         skip_unlabeled=True,                    # skip trajectories without language labels
    #         goal_relabeling_strategy="uniform",     # goals are currently unused
    #     ),
    #     frame_transform_kwargs=dict(
    #         resize_size=resize_size,
    #         num_parallel_calls=16,                  # for the most CPU-intensive ops (decoding, resizing, augmenting)
    #     ),
    #     dataset_kwargs_list=data_kwargs_list,
    #     shuffle_buffer_size=1000,
    #     sample_weights=weights,
    #     balance_weights=True,
    #     traj_transform_threads=len(vla_data_mix),
    #     traj_read_threads=len(vla_data_mix),
    #     train=True,
    # )
    # dataset = LLaVARLDSDataset(
    #     llava_transform = transform,
    #     **data_config
    # )
    # collator = PaddedCollatorForLanguageModeling(
    #     tokenizer.model_max_length,
    #     tokenizer.pad_token_id,
    #     vision_backbone.default_image_resolution,
    #     padding_side=cfg.padding_side_override,
    # )

    # Get VLA Dataset & Collator
    print(f"Creating VLA Open-X Dataset with Mixture `{cfg.vla.data_mix}`")
    image_transform = vlm.vision_backbone.get_image_transform()
    vla_dataset, action_tokenizer, collator = get_vla_dataset_and_collator(
        cfg.data_root_dir,
        cfg.vla.data_mix,
        image_transform=image_transform,
        tokenizer=vlm.llm_backbone.get_tokenizer(),
        prompt_builder_fn=vlm.llm_backbone.prompt_builder_fn,
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
        shuffle_buffer_size=cfg.vla.shuffle_buffer_size,
    )
    # Create dataloader
    batch_size = 1
    dataloader = DataLoader(
        vla_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
        shuffle=False,
    )
    # Prepare aggregate metrics dict
    stats_dict = {
        "teacher_forcing": {},
        "no_teacher_forcing": {},
        "no_teacher_forcing_manual_generate": {},
        "teacher_forcing_prompt_builder": {},
    }
    for k in stats_dict.keys():
        stats_dict[k]["l1_loss"] = 0.
        stats_dict[k]["actions_accuracy"] = 0.
        stats_dict[k]["avg_l1_loss"] = 0.
        stats_dict[k]["avg_actions_accuracy"] = 0.
    cnt = 0
    for idx, batch in enumerate(dataloader):
        print(f"===========================================================")
        print(f"Batch {idx}:")
        print(f"===========================================================")
        # Teacher forcing
        actions_accuracy, l1_loss = eval_teacher_forcing(batch, vlm, action_tokenizer, device)
        stats_dict["teacher_forcing"]["l1_loss"] += l1_loss
        stats_dict["teacher_forcing"]["actions_accuracy"] += actions_accuracy
        print(f"-----------------------------------------------------------")
        # # No teacher forcing: via `generate()`
        # actions_accuracy, l1_loss = eval_no_teacher_forcing(batch, vlm, action_tokenizer, device)
        # stats_dict["no_teacher_forcing"]["l1_loss"] += l1_loss
        # stats_dict["no_teacher_forcing"]["actions_accuracy"] += actions_accuracy
        # print(f"-----------------------------------------------------------")
        # # No teacher forcing: via manual autoregressive sequential output prediction
        # actions_accuracy, l1_loss = eval_no_teacher_forcing_manual_generate(batch, vlm, action_tokenizer, device)
        # stats_dict["no_teacher_forcing_manual_generate"]["l1_loss"] += l1_loss
        # stats_dict["no_teacher_forcing_manual_generate"]["actions_accuracy"] += actions_accuracy
        # print(f"-----------------------------------------------------------")
        # # No teacher forcing (alt)
        # if batch_size == 1:
        #     actions_accuracy, l1_loss = eval_no_teacher_forcing_prompt_builder(batch, vlm, action_tokenizer, tokenizer, image_transform, device)
        #     stats_dict["teacher_forcing_prompt_builder"]["l1_loss"] += l1_loss
        #     stats_dict["teacher_forcing_prompt_builder"]["actions_accuracy"] += actions_accuracy
        cnt += 1
        if cnt == 100:
            break
    # Compute aggregrate metrics
    for k in stats_dict.keys():
        stats_dict[k]["avg_l1_loss"] = stats_dict[k]["l1_loss"] / cnt
        stats_dict[k]["avg_actions_accuracy"] = stats_dict[k]["actions_accuracy"] / cnt
    print(f"===========================================================")
    print(f"Aggregate metrics:")
    print(f"===========================================================")
    print(f"# batches: {cnt}")
    for k in stats_dict.keys():
        if k == "teacher_forcing":
            print(f"Teacher forcing:")
        elif k == "no_teacher_forcing":
            print(f"No teacher forcing - generate():")
        elif k == "no_teacher_forcing_manual_generate":
            print(f"No teacher forcing - manual autoregressive generation:")
        else:
            print(f"No teacher forcing - prompt builder:")
        print(f"    Average L1 loss: {stats_dict[k]['avg_l1_loss']}")
        print(f"    Average action tokens accuracy: {stats_dict[k]['avg_actions_accuracy']}")


if __name__ == "__main__":
    eval()
