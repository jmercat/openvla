"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from prismatic.conf import ModelConfig
from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlms import OpenVLA, PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")

        # Get paths for `config.json` and pretrained checkpoint
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        with overwatch.local_zero_first():
            config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
            checkpoint_pt = hf_hub_download(
                repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
            )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Loading Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=not load_for_training,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        freeze_weights=not load_for_training,
    )

    return vlm


# === Load Pretrained VLA Model ===
def load_vla(
    model_path: Union[str, Path],
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    load_for_training: bool = False,
) -> OpenVLA:
    """Loads a pretrained VLA model directly from checkpoint path."""
    overwatch.info(f"Loading from local checkpoint path `{model_path}`")

    # Assert that the checkpoint path looks like: `..../<RUN_ID>/checkpoints/<CHECKPOINT_DIR>`
    model_path = str(model_path)
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
        norm_stats = json.load(f)

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
    vla = OpenVLA.from_pretrained(
        checkpoint_pt,
        model_cfg.model_id,
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg.arch_specifier,
        freeze_weights=not load_for_training,
        norm_stats=norm_stats,
        action_tokenizer=action_tokenizer,
    )

    return vla
