"""
vla.py

Draccus Dataclass Definition for a VLAConfig object, with various registered subclasses for each VLA experiment and
model configuration thereof. A given VLA model (`policy`) configures the following attributes:
    - Data Mixture (e.g., Bridge, OXE_MAGIC_SOUP, etc.)
    - Base VLM from Prismatic Registry (e.g., `prism-dinosiglip+7b`)
    - VLA Model Architecture / Parameters (e.g., freeze vision encoder, last layer finetuning)
    - Training / Optimization Hyperparameters
"""

from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Optional, Union

from draccus import ChoiceRegistry


@dataclass
class VLAConfig(ChoiceRegistry):
    # fmt: off
    vla_id: str                                     # Unique VLA Policy ID that fully specifies a configuration variant
    base_vlm: Union[str, Path]                      # Base VLM as ID/Path to Run Directory (e.g., `prism-dinosiglip+7b`)
    freeze_vision_backbone: bool                    # Freeze Vision Backbone Parameters (akin to pretraining)
    freeze_llm_backbone: bool                       # Freeze LLM Backbone parameters
    unfreeze_last_llm_layer: bool                   # Unfreeze final layer of LLM (only takes effect if LLM is frozen)

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

    # Model Parameters
    action_chunk_length: int                        # Length of action chunks predicted by the model

    # Optimization Parameters
    epochs: int                                     # Epochs to Run (in case `max_steps` is not specified)
    max_steps: Optional[int]                        # [Optional] Max Gradient Steps to Run (overrides `epochs`)

    expected_world_size: int                        # Expected # of GPUs =>> allows us to gate training on hardware
    global_batch_size: int                          # Global Batch Size (divided across processes / world size)
    per_device_batch_size: int                      # Per-Device Batch Size (per-process / individual GPU)
                                                    #   =>> # of accumulation steps is auto-computed

    learning_rate: float                            # Peak Learning Rate (`lr_scheduler_type` sets warmup/decay)
    weight_decay: float                             # Weight Decay for AdamW Optimizer
    max_grad_norm: float                            # Max Grad Norm (for global gradient clipping)
    lr_scheduler_type: str                          # LR Scheduler (usually: "constant" | "linear-warmup+cosine-decay")
    warmup_ratio: float                             # Fraction of Steps to Warmup (for warmup LR schedulers)

    train_strategy: str                             # Train Strategy (default "fsdp-full-shard")

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True      # Enable Gradient/Activation Checkpointing during Training

    # Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True    # Enable Traditional BF16 Mixed Precision
    reduce_in_full_precision: bool = True           # Accumulate/Reduce All-Gather Gradients in FP32 Full Precision

    # fmt: on


# === [8 GPU] Base VLA =>> LLaVa (Reproduction) + Bridge ===
@dataclass
class Exp_LLaVa15_Bridge(VLAConfig):
    vla_id: str = "reproduction-llava-v15+mx-bridge"
    base_vlm: Union[str, Path] = "reproduction-llava-v15+7b"
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = False
    unfreeze_last_llm_layer: bool = False

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 256_000

    # Model Parameters
    action_chunk_length: int = 1

    # Optimization Parameters
    epochs: int = 1000
    max_steps: Optional[int] = None

    expected_world_size: int = 8
    global_batch_size: int = 256
    per_device_batch_size: int = 32

    learning_rate: float = 5e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "constant"
    warmup_ratio: float = 0.0

    train_strategy: str = "fsdp-full-shard"


# === [8 GPU] Fast Iteration =>> SigLIP 224px + Bridge ===
@dataclass
class Exp_SigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"


# === [8 GPU] SigLIP 224px Frozen Vision Backbone + Bridge ===
@dataclass
class Exp_FreezeVIT_SigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px-icy+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True


# === [8 GPU] LLaVa (Reproduction) Frozen Vision Backbone + Bridge ===
@dataclass
class Exp_FreezeVIT_LLaVa15_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "reproduction-llava-v15-icy+mx-bridge"
    freeze_vision_backbone: bool = True


# === [8 GPU] Best Prism 7B Model =>> DINO-SigLIP @ 384px + Bridge ===
@dataclass
class Exp_DINOSigLIP_384px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "prism-dinosiglip+mx-bridge"
    base_vlm: Union[str, Path] = "prism-dinosiglip+7b"

    # Note =>> Unfrozen DINOSigLIP OOMs w/ Per-Device Batch Size of 32!
    global_batch_size: int = 192
    per_device_batch_size: int = 24


# === [8 GPU] Frozen Vision Backbone =>> DINO-SigLIP @ 384px + Bridge ===
@dataclass
class Exp_FreezeVIT_DINOSigLIP_384px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "prism-dinosiglip-icy+mx-bridge"
    base_vlm: Union[str, Path] = "prism-dinosiglip+7b"
    freeze_vision_backbone: bool = True

    # Note =>> Frozen DINOSigLIP can Handle Per-Device Batch Size of 32
    #   - HOWEVER :: For fair comparison with "Unfrozen" --> run with lower batch size!
    global_batch_size: int = 192
    per_device_batch_size: int = 24


# === [8 GPU] First Attempt LR Sweep w/ SigLIP 224px + Unfrozen Backbone + Bridge ===
@dataclass
class Exp_LR1E5_SigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "lr-1e5+siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    learning_rate: float = 1e-5


@dataclass
class Exp_LR2E5_SigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "lr-2e5+siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    learning_rate: float = 2e-5


@dataclass
class Exp_LR4E5_SigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "lr-4e5+siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    learning_rate: float = 4e-5


@dataclass
class Exp_LR1E4_SigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "lr-1e4+siglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    learning_rate: float = 1e-4


# === [64 GPU] SigLIP 224px + OXE Magic Soup ===
@dataclass
class Exp_SigLIP_224px_OXE_Magic_Soup(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-oxe-magic-soup"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "oxe_magic_soup"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5


# === [8 GPU] Fast Iteration =>> SigLIP 224px + DROID ===
@dataclass
class Exp_SigLIP_224px_DROID(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-droid"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "droid"


# === [8 GPU] Fast Iteration =>> DINO-SigLIP 224px + Bridge ===
@dataclass
class Exp_DinoSigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-bridge"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    data_mix: str = "bridge"
    learning_rate: float = 2e-5


# === [64 GPU] DINO-SigLIP 224px + OXE Magic Soup++ ===
@dataclass
class Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus(Exp_LLaVa15_Bridge):
    vla_id: str = "prism-dinosiglip-224px+mx-oxe-magic-soup-plus"
    base_vlm: Union[str, Path] = "prism-dinosiglip-224px+7b"

    # data_mix: str = "oxe_magic_soup_plus"
    data_mix: str = "oxe_magic_soup_plus_minus"

    expected_world_size: int = 64
    global_batch_size: int = 2048
    per_device_batch_size: int = 32

    learning_rate: float = 2e-5


# === [8 GPU] SigLIP 224px + LIBERO-Spatial ===
@dataclass
class Exp_SigLIP_224px_LIBERO_Spatial(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-libero_spatial"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "libero_spatial"
    learning_rate: float = 2e-5


# === [8 GPU] SigLIP 224px + T-DROID ===
@dataclass
class Exp_SigLIP_224px_Tdroid_CarrotInBowl(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_carrot_in_bowl"
    learning_rate: float = 2e-5


@dataclass
class Exp_SigLIP_224px_Tdroid_PourCornInPot(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-tdroid_pour_corn_in_pot"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "tdroid_pour_corn_in_pot"
    learning_rate: float = 2e-5


# === [8 GPU] SigLIP 224px + T-DROID -- Partial Finetuning ===
@dataclass
class Exp_SigLIP_224px_Icy_Tdroid_CarrotInBowl(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px-icy+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = False

    data_mix: str = "tdroid_carrot_in_bowl"
    learning_rate: float = 2e-5


@dataclass
class Exp_SigLIP_224px_LastLayer_Tdroid_CarrotInBowl(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px-last_layer+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"
    learning_rate: float = 2e-5


@dataclass
class Exp_SigLIP_224px_Sandwich_Tdroid_CarrotInBowl(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px-sandwich+mx-tdroid_carrot_in_bowl"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = False
    freeze_llm_backbone: bool = True
    unfreeze_last_llm_layer: bool = True

    data_mix: str = "tdroid_carrot_in_bowl"
    learning_rate: float = 2e-5


# === [8 GPU] SigLIP 224px + FrankaWipe ===
@dataclass
class Exp_SigLIP_224px_Droid_Wipe(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-droid_wipe"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "droid_wipe"
    learning_rate: float = 2e-5


@dataclass
class Exp_SigLIP_224px_Droid_Wipe_Chunk16(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-droid_wipe-chunk16"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "droid_wipe"
    learning_rate: float = 2e-5
    action_chunk_length: int = 16


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    LLAVA_REPRO_MX_BRIDGE = Exp_LLaVa15_Bridge
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge

    # Initial SigLIP Frozen Backbone Experiment
    FREEZE_SIGLIP_224PX_MX_BRIDGE = Exp_FreezeVIT_SigLIP_224px_Bridge

    # [03/12] Additional Frozen Backbone Experiments + DINOSigLIP + SigLIP LR Sweep (Shallow)
    FREEZE_LLAVA_REPRO_MX_BRIDGE = Exp_FreezeVIT_LLaVa15_Bridge

    DINOSIGLIP_384PX_MX_BRIDGE = Exp_DINOSigLIP_384px_Bridge
    FREEZE_DINOSIGLIP_384PX_MX_BRIDGE = Exp_FreezeVIT_DINOSigLIP_384px_Bridge

    LR_1E5_SIGLIP_224PX_MX_BRIDGE = Exp_LR1E5_SigLIP_224px_Bridge
    LR_2E5_SIGLIP_224PX_MX_BRIDGE = Exp_LR2E5_SigLIP_224px_Bridge
    LR_4E5_SIGLIP_224PX_MX_BRIDGE = Exp_LR4E5_SigLIP_224px_Bridge
    LR_1E4_SIGLIP_224PX_MX_BRIDGE = Exp_LR1E4_SigLIP_224px_Bridge

    # [03/21] OXE Magic Soup Run
    SIGLIP_224PX_MX_OXE_MAGIC_SOUP = Exp_SigLIP_224px_OXE_Magic_Soup

    # [03/28] DROID Experiments
    SIGLIP_224PX_MX_DROID = Exp_SigLIP_224px_DROID

    # [05/07] DinoSiglip + Bridge
    DINOSIGLIP_224PX_MX_BRIDGE = Exp_DinoSigLIP_224px_Bridge

    # [04/18] OXE Magic Soup Plus Run
    DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS = Exp_DinoSigLIP_224px_OXE_Magic_Soup_Plus

    # [04/25] Libero Run
    SIGLIP_224PX_MX_LIBERO_SPATIAL = Exp_SigLIP_224px_LIBERO_Spatial

    # [05/08] T-DROID Runs
    SIGLIP_224PX_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Tdroid_CarrotInBowl
    SIGLIP_224PX_MX_TDROID_POUR_CORN_IN_POT = Exp_SigLIP_224px_Tdroid_PourCornInPot

    # [05/08] T-DROID Partial Finetuning Runs
    SIGLIP_224PX_ICY_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Icy_Tdroid_CarrotInBowl
    SIGLIP_224PX_LASTLAYER_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_LastLayer_Tdroid_CarrotInBowl
    SIGLIP_224PX_SANDWICH_MX_TDROID_CARROT_IN_BOWL = Exp_SigLIP_224px_Sandwich_Tdroid_CarrotInBowl

    # [05/14] DROID Wipe Runs
    SIGLIP_224PX_MX_DROID_WIPE = Exp_SigLIP_224px_Droid_Wipe
    SIGLIP_224PX_MX_DROID_WIPE_CHUNK16 = Exp_SigLIP_224px_Droid_Wipe_Chunk16

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
