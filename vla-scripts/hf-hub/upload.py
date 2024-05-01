"""
upload.py

Utility script for uploading VLA checkpoints to the HuggingFace Hub, under the `openvla/openvla-dev` model repository.
Makes it easy to share/load arbitrary checkpoints via `hf_hub_download` (with built-in caching logic).

Preliminaries:
    - Install the Rust-based HF-Transfer Extension --> `pip install --upgrade huggingface_hub[hf_transfer]`
        - Enable: `export HF_HUB_ENABLE_HF_TRANSFER=1`
    - Login via the HuggingFace CLI --> `huggingface-cli login`
    - Verify that `openvla` is in "orgs" --> `huggingface-cli whoami`

Run with: `python vla-scripts/hf-hub/upload.py \
    --run_dir="/mnt/fsx/.../siglip-224px+mx-oxe-magic-soup+n8+b32+x7" \
    --steps_to_upload="[152500, 155000]"
"""

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import draccus


@dataclass
class UploadConfig:
    # fmt: off
    run_dir: Union[str, Path]               # Absolute path to top-level run directory (should contain `config.yaml`)
    steps_to_upload: List[int] = field(     # Checkpoint steps to upload (subset of all checkpoints saved)
        default_factory=lambda: [152500, 155000]
    )

    model_type: str = "pretrained"          # Model type in < pretrained | finetuned >

    hub_repo: str = "openvla/openvla-dev"   # HF Hub Repository

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)

        # Validate
        assert self.run_dir.exists() and (self.run_dir / "config.yaml").exists(), "Missing config.yaml in `run_dir`!"

    # fmt: on


@draccus.wrap()
def upload(cfg: UploadConfig) -> None:
    print(f"[*] Uploading from `{cfg.run_dir}` to HF-Hub :: `{cfg.hub_repo}`")

    # Set HF Hub relative path
    hub_relpath = Path(f"{cfg.model_type}/{cfg.run_dir.name}/")

    # Upload all top-level files (config & metrics)
    print("\n[*] Uploading Top-Level Config & Metrics Files")
    upload_files = [fn for fn in cfg.run_dir.iterdir() if fn.is_file()]
    for fn in upload_files:
        subprocess.run(
            f"huggingface-cli upload {cfg.hub_repo} {fn!s} {(hub_relpath / fn.name)!s}", shell=True, check=True
        )

    # Upload selected checkpoints
    print("\n[*] Uploading Selected Checkpoints (will take a while...)")
    checkpoint_dir = cfg.run_dir / "checkpoints"
    for step in cfg.steps_to_upload:
        print(f"\t=>> Uploading Step {step} Checkpoint")
        matches = list(checkpoint_dir.glob(f"*{step:06d}*"))
        assert len(matches) == 1, f"Found more than one checkpoint for step {step} =>> `{matches}`!"

        # Upload
        fn = matches[0]
        subprocess.run(
            f"huggingface-cli upload {cfg.hub_repo} {fn!s} {(hub_relpath / 'checkpoints' / fn.name)!s}",
            shell=True,
            check=True,
        )

    # Done!
    print(f"\n[*] Done =>> Check https://huggingface.co/{cfg.hub_repo}/tree/main/{cfg.model_type}/{cfg.run_dir.name}")


if __name__ == "__main__":
    upload()
