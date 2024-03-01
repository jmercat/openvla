# Prismatic - Vision-Language-Action Models for Robotics

Research and development codebase for training visually-conditioned language-models (VLMs) and vision-language-action
models (VLAs). Built on top of [TRI-ML/prismatic-vlms](https://github.com/TRI-ML/prismatic-vlms).

## 🚨 Note for Collaborators

This VLA codebase is functionally a private fork of the open-source `TRI-ML/prismatic-vlms` repository. To
facilitate a clean workflow with the open-source VLM codebase (and any additional features added), we adopt the
following structure:

- **[Default]** `vla-core` - Treat this as the `main` branch for developing any new VLA changes; always PR to this
  branch in lieu of `main`.
- `vlm-core` - This is the central branch for developing new VLM features (that are meant to be pushed to the public
  open-source code). Sidd/Suraj will sync upstream changes to `vla-core`.
- `main` - Treat this as a *locked branch*; it tracks the latest stable code in the open-source VLM repository.

#### Default Setup Instructions

*Note: TRI folks should follow the [TRI Setup Instructions](#tri-setup-instructions) below!*

Fork this repository to your personal account (e.g., `moojink/prismatic-dev`). This will automatically set `vla-core`
as your main working branch. Set up your remotes to track this repository `siddk/prismatic-dev`:

```bash
# This should indicate that `origin` is set to your local fork (e.g., `moojink/prismatic-dev.git`)
git remote -v

# Add `siddk/prismatic-dev.git` as a separate remote (conventionally `upstream`; I prefer `sk-origin`)
git remote add sk-origin https://github.com/siddk/prismatic-dev.git

# [Periodically] Sync any upstream changes to your local branch
git pull sk-origin vla-core
```

Cut a new (local) feature branch for anything you want to add to the Prismatic codebase:

```bash
# Create a new (local) feature branch after syncing `vla-core`
git switch -c <feature-branch-name>

# Do work... commit frequently...
git add <changed files>
git commit -m "<informative and clean commit message>"

# Push to *local* fork (`origin`)
git push -u origin <feature-branch-name>
```

When ready, initiate PR to `siddk/prismatic-dev@vla-core`. The maintainers (Sidd/Moo Jin/Suraj/Karl) will review and
merge into `vla-core`.


#### TRI Setup Instructions

For TRI collaborators, the above process is a bit different, as you should already have an internal, local fork of the
`TRI-ML/prismatic-dev` codebase (with `vlm-core` as default branch). To contribute to the VLA codebase, do the
following:

```bash
# Switch to `vla-core` on your local branch (e.g., `suraj-nair-tri/prismatic-dev@vla-core`)
git checkout vla-core

# This should indicate `origin` is set to your local fork (e.g., `suraj-nair-tri/prismatic-dev.gt`) AND that
# `tri-origin` is set to the TRI internal repo (e.g., `TRI-ML/prismatic-dev.git`).
git remote -v

# Add `siddk/prismatic-dev.git` as a separate remote (conventionally `upstream`; I prefer `sk-origin`)
#   => After this step, you'll have 3 remotes (`sk-origin`, `tri-origin`, and your local `origin`)
git remote add sk-origin https://github.com/siddk/prismatic-dev.git

# Treat `sk-origin` as the source of truth for `vla-core` - periodically sync
git pull sk-origin vla-core
```

When contributing, just make sure to PR to `siddk/prismatic-dev@vla-core` **not** the TRI-ML repository. Sidd/Suraj
will handle keeping things in sync (including any changes to `vlm-core`).

---

## Installation

This repository was built using Python 3.10, but should be backwards compatible with any Python >= 3.8. We require
PyTorch 2.1 or greater installation instructions [can be found here](https://pytorch.org/get-started/locally/). This
repository was developed and has been thoroughly tested with:
  - [2/16/24] PyTorch 2.1.0, Torchvision 0.16.0, Transformers 4.34.1, and Flash-Attention 2.3.3.
  - [2/24/24] PyTorch 2.2.1, Torchvision 0.17.0, Transformers 4.38.1, and Flash-Attention 2.5.5.

Once PyTorch has been properly installed (e.g., via
`conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`), you can install this
package locally via an editable installation:

```bash
cd prismatic-dev
pip install -e ".[dev]"
pre-commit install

# Training additionally requires Flash-Attention 2 (https://github.com/Dao-AILab/flash-attention)
pip install packaging ninja

# Verify Ninja --> should return exit code "0"
ninja --version; echo $?

# Install Flash Attention 2
#   =>> If you run into difficulty, try `pip cache remove flash_attn` first
pip install flash-attn --no-build-isolation
```

If you run into any problems during the installation process, please file a GitHub Issue.

## Prismatic VLM Usage

Once installed, loading and running inference with pretrained `prismatic` VLMs is easy:

```python
import requests
import torch

from PIL import Image
from pathlib import Path

from prismatic import load

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
hf_token = Path(".hf_token").read_text().strip()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub)
model_id = "prism-dinosiglip+7b"
vlm = load(model_id, hf_token=hf_token)
vlm.to(device, dtype=torch.bfloat16)

# Download an image and specify a prompt
image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
user_prompt = "What is going on in this image?"

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=1,
)
```

For a complete terminal-based CLI for interacting with our VLMs, check out [scripts/generate.py](scripts/generate.py).

## Pretrained VLMs

We release **all 42** VLMs trained as part of our work, with a range of different visual representations, language
models, data, and scale. The exhaustive set of models (with structured descriptions) can be found in
[`prismatic/models/registry.py](prismatic/models/registry.py) - we will continue to update this registry as we train
additional models.

We also provide a top-level API for instantiating models from the names mentioned in the various Figures of our paper,
as well as for generally browsing our pretrained models by description:

```python
from prismatic import available_model_names, available_models, get_model_description
from pprint import pprint

# List all Pretrained VLMs (by HF Hub IDs)
pprint(available_models())

# List all Pretrained VLMs + Descriptions (by explicit labels / names from paper figures)
pprint(available_model_names())

# Print and return a targeted description of a model (by name or ID)
#   =>> See `prismatic/models/registry.py` for explicit schema
description = get_model_description("Prism-DINOSigLIP 13B (Controlled)")
```

Currently, our best performing models are the `Prism-DINOSigLIP` series, with especially strong performance on spatial
understanding and localization tasks.

---
**Explicit Notes on Model Licensing & Commercial Use**: While all code in this repository is released under an MIT
License, our pretrained models inherit restrictions from the _datasets_ and _underlying LMs_ we use for training.

**[02/09/24]** Our current VLMs are all derived from Llama-2, and as such are subject to the
[Llama Community License](https://ai.meta.com/llama/license/), which does permit commercial use. We additionally train
on the LLaVa Instruct Tuning data, which is synthetically generated using OpenAI's GPT-4 (subject to the
[OpenAI Terms of Use](https://openai.com/policies/terms-of-use)).

As we train new models, we will update this section of the README (and the LICENSE files associated with each model)
appropriately. If there are any questions, please file an Issue!

## Training VLMs

*Note: This section describes training VLMs — not VLA models for robotics. For VLA training, see
[Training VLAs](#training-vlas).*

In addition to providing all pretrained VLMs trained in this work, we also provide full instructions and configurations
for _reproducing all results_ (down to controlling for the batch order of examples seen during training).

#### Pretraining Datasets

For the [LLaVa v1.5 Instruct Dataset](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md) we use for all
of our models, we provide an automated download script in [`scripts/preprocess.py`](scripts/preprocess.py):

```bash
# Download the `llava-v1.5-instruct` (Instruct Tuning) Image and Language Data (includes extra post-processing)
python scripts/preprocess.py --dataset_id "llava-v1.5-instruct" --root_dir <PATH-TO-DATA-ROOT>

# (In case you also wish to download the explicit vision-language alignment data)
python scripts/preprocess.py --dataset_id "llava-laion-cc-sbu-558k" --root_dir <PATH-TO-DATA-ROOT>
```

As part of our work, we also train on mixtures of datasets including
[LVIS-Instruct-4V](https://arxiv.org/abs/2311.07574) and [LRV-Instruct](https://arxiv.org/abs/2306.14565). We provide
instructions and scripts for downloading these datasets in [`scripts/additional-datasets`](scripts/additional-datasets).

We welcome any and all contributions and pull requests to add new datasets!

#### Model Configuration & Training Script

The entry point for training models is [`scripts/pretrain.py`](scripts/pretrain.py). We employ
[`draccus`](https://pypi.org/project/draccus/0.6/) to provide a modular, dataclass-based interface for specifying
model configurations; all 42 VLM configurations are in [`prismatic/conf/models.py`](prismatic/conf/models.py).

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs, though we also provide a simpler
Distributed Data Parallel training implementation (for smaller LM backbones, debugging). You can run a pretraining job
via `torchrun`.

As a compact example, here's how you would train a VLM derived from Vicuña-v1.5 7B, using fused DINOv2 + SigLIP
representations, processing non-square images with a "letterbox padding" transform across 8 GPUs on a single-node:

```bash
# Run from the root of the repository
torchrun --standalone --nnodes 1 --nproc-per-node 8 scripts/pretrain.py \
  --model.type "one-stage+7b" \
  --model.model_id "<NAME OF NEW MODEL>" \
  --model.vision_backbone_id "dinosiglip-vit-so-384px" \
  --model.image_resize_strategy "letterbox" \
  --model.llm_backbone_id "vicuna-v15-7b"
```

Note that specifying `model.type` is important for identifying the _base configuration_ that you want to build on top of;
the full list of model types are available in our [config file](prismatic/conf/models.py), under the `model_id` key for
each dataclass.

---

## Training VLAs

We provide full instructions and configurations for training different VLA policies on (arbitrary subsets of) the
[Open-X Embodiment (OXE) Dataset](https://robotics-transformer-x.github.io/). Setup instructions are the same as above (see
[Installation](#installation)), but if you run into any issues, see [VLA Troubleshooting](#vla-troubleshooting) below.

#### VLA Pretraining Datasets

For the bulk of our experiments, we use [the `OXE_MAGIC_SOUP` tagged mixture](https://github.com/siddk/prismatic-dev/blob/vla-core/prismatic/vla/datasets/rlds/oxe/mixtures.py#L112)
of Open-X Embodiment. We download and preprocess them in [RLDS format](https://github.com/google-research/rlds)
following [Karl's custom script](https://github.com/kpertsch/rlds_dataset_mod/blob/main/prepare_open_x.sh).
- **Important**: For the Bridge V2 component dataset, the version in OXE is out of date (as of 12/20/2023). Instead,
  you should download the dataset from the
  [official website](https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/) and place it
  under the subdirectory `bridge_orig/`. Replace any reference to `bridge` in the OXE code with `bridge_orig`.

#### VLA Configuration & Training Script

The entry point for VLA training is [`vla-scripts/train.py`](vla-scripts/train.py). We employ
[`draccus`](https://pypi.org/project/draccus/0.6/) to provide a modular, dataclass-based interface for specifying
VLA configurations; existing VLA configurations are in [`prismatic/conf/vla.py`](prismatic/conf/vla.py). You can add
your own training configuration and refer to it using the `--vla.type` command line argument.

We use PyTorch Fully Sharded Data Parallel (FSDP) to distribute training across GPUs. Launch training jobs via
`torchrun`:

```bash
# Train VLA on Bridge V2 with the Prismatic SigLIP 224px Backbone on a Single Node (w/ 8 GPUs)
#   => Note: To log / access the `openvla` project under the `stanford-voltron` entity, ask Sidd/Suraj/Moo Jin!
torchrun --standalone --nnodes 1 --nproc-per-node 8 vla-scripts/train.py \
  --vla.type "siglip-224px+mx-bridge" \
  --data_root_dir <PATH TO OXE DATA ROOT> \
  --run_root_dir <PATH TO LOG/CHECKPOINT ROOT> \
  --wandb_project "openvla" \
  --wandb_entity "stanford-voltron"
```

#### VLA Troubleshooting

The following are a list of known problems and corresponding fixes:

```bash
FileNotFoundError: Failed to construct dataset "fractal20220817_data", builder_kwargs "{'data_dir': '/path/to/processed/datasets/'}": Could not load dataset info from fractal20220817_data/0.1.0/dataset_info.json
```
- **Fix**: Downgrade `tensorflow-datasets` via `pip install tensorflow-datasets==4.9.3`.


```bash
AttributeError: 'DLataset' object has no attribute 'traj_map'. Did you mean: 'flat_map'?
```
- **Fix**: Upgrade `dlimp` to the newest version. You may have to `--force-reinstall` like so:
`pip install --no-deps --force-reinstall git+https://github.com/kvablack/dlimp@ad72ce3a9b414db2185bc0b38461d4101a65477a`

---

## Repository Structure

High-level overview of repository/project file-tree:

+ `prismatic` - Package source; provides core utilities for model loading, training, data preprocessing, etc.
+ `scripts/` - Standalone scripts for preprocessing, training VLMs, and generating from pretrained models.
+ `vla-scripts/` - Standalone scripts for training and evaluating VLA models for robotics.
+ `LICENSE` - All code is made available under the MIT License; happy hacking!
+ `Makefile` - Top-level Makefile (by default, supports linting - checking & auto-fix); extend as needed.
+ `pyproject.toml` - Full project configuration details (including dependencies), as well as tool configurations.
+ `README.md` - You are here!
