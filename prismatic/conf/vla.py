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

    # Data Mixture Parameters
    data_mix: str                                   # Open-X Embodiment Dataset =>> Unique Mixture ID (e.g., `bridge`)
    shuffle_buffer_size: int                        # Size of Shuffle Buffer (100K for Bridge, 1M for OXE)

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

    # Data Mixture Parameters
    data_mix: str = "bridge"
    shuffle_buffer_size: int = 256_000

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


# === [16 GPU] Bridge + RT-1 =>> Unfrozen SigLIP 224px + [Bridge, RT-1] ===
@dataclass
class Exp_SigLIP_224px_Bridge_RT1(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px+mx-bridge-rt1"
    base_vlm: Union[str, Path] = "siglip-224px+7b"

    data_mix: str = "bridge_rt_1"

    expected_world_size: int = 16
    global_batch_size: int = 512


# === [32 GPU] Bridge + RT-1 =>> Frozen DINO-SigLIP @ 384px + [Bridge, RT-1] ===
@dataclass
class Exp_FreezeVIT_DINOSigLIP_384px_Bridge_RT1(Exp_LLaVa15_Bridge):
    vla_id: str = "prism-dinosiglip-icy+mx-bridge-rt1"
    base_vlm: Union[str, Path] = "prism-dinosiglip+7b"
    freeze_vision_backbone: bool = True

    data_mix: str = "bridge_rt_1"

    expected_world_size: int = 32
    global_batch_size: int = 1024


# === [8 GPU] LLaVa (Reproduction) Frozen Vision Backbone + Bridge ===
@dataclass
class Exp_FreezeVIT_LLaVa15_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "reproduction-llava-v15-icy+mx-bridge"
    freeze_vision_backbone: bool = True


# === [8 GPU] SigLIP 224px Frozen Vision Backbone + Bridge ===
@dataclass
class Exp_FreezeVIT_SigLIP_224px_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "siglip-224px-icy+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True


# === [8 GPU] First Attempt LR Sweep w/ SigLIP 224px + Frozen Backbone + Bridge ===
@dataclass
class Exp_LR1E5_SigLIP_224px_Icy_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "lr-1e5+siglip-224px-icy+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True

    learning_rate: float = 1e-5


@dataclass
class Exp_LR2E5_SigLIP_224px_Icy_Bridge(Exp_LLaVa15_Bridge):
    vla_id: str = "lr-2e5+siglip-224px-icy+mx-bridge"
    base_vlm: Union[str, Path] = "siglip-224px+7b"
    freeze_vision_backbone: bool = True

    learning_rate: float = 2e-5


# === Define a VLA Registry Enum for Reference & Validation ===
@unique
class VLARegistry(Enum):
    LLAVA_REPRO_MX_BRIDGE = Exp_LLaVa15_Bridge
    SIGLIP_224PX_MX_BRIDGE = Exp_SigLIP_224px_Bridge

    # [2/25] SK Initial Spread =>> DINOSigLIP, Freeze Vision Backbone, Multi-Node Bridge + RT-1
    DINOSIGLIP_384PX_MX_BRIDGE = Exp_DINOSigLIP_384px_Bridge
    FREEZE_DINOSIGLIP_384PX_MX_BRIDGE = Exp_FreezeVIT_DINOSigLIP_384px_Bridge

    SIGLIP_224PX_MX_BRIDGE_RT1 = Exp_SigLIP_224px_Bridge_RT1
    FREEZE_DINOSIGLIP_384PX_MX_BRIDGE_RT1 = Exp_FreezeVIT_DINOSigLIP_384px_Bridge_RT1

    # [3/03] Additional Frozen Backbone Experiments + SigLIP LR Sweep (Shallow)
    FREEZE_LLAVA_REPRO_MX_BRIDGE = Exp_FreezeVIT_LLaVa15_Bridge
    FREEZE_SIGLIP_224PX_MX_BRIDGE = Exp_FreezeVIT_SigLIP_224px_Bridge

    LR_1E5_SIGLIP_224PX_ICY_MX_BRIDGE = Exp_LR1E5_SigLIP_224px_Icy_Bridge
    LR_2E5_SIGLIP_224PX_ICY_MX_BRIDGE = Exp_LR2E5_SigLIP_224px_Icy_Bridge

    @property
    def vla_id(self) -> str:
        return self.value.vla_id


# Register VLAs in Choice Registry
for vla_variant in VLARegistry:
    VLAConfig.register_subclass(vla_variant.vla_id, vla_variant.value)
