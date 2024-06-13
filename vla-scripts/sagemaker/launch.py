"""
launch.py

Utility script for launch VLA training jobs (`scripts/train.py`) via Sagemaker, with multi-node support.

Run with: `python vla-scripts/sagemaker/launch.py <ARGS>`
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import draccus
import sagemaker
import wandb
from sagemaker.inputs import FileSystemInput
from sagemaker.pytorch import PyTorch

from prismatic.conf import VLARegistry

# === Constants ===
ROLE_ARN = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess"
SUBNETS = ["subnet-07bf42d7c9cb929e4", "subnet-05f1115c7d6ccbd07", "subnet-0e260ba29726b9fbb"]
SECURITY_GROUP_IDS = ["sg-0afb9fb0e79a54061", "sg-0333993fea1aeb948", "sg-0c4b828f4023a04cc"]

S3_LOG_PATH = "s3://tri-ml-sandbox-16011-us-east-1-datasets/sagemaker/prismatic-vlas/"
LUSTRE_PARAMETERS = {
    "file_system_type": "FSxLustre",
    "file_system_access_mode": "rw",
    "file_system_id": "fs-0ee5fb54e88f9dd00",
    "directory_path": "/kxvmdbev",
}


@dataclass
class LaunchConfig:
    # fmt: off
    job_name: str = "sk-openvla"                                        # Base Name for Job in Sagemaker Dashboard
    instance_count: int = 8                                             # Number of Nodes for Multi-Node Training
    instance_type: str = "ml.p4de.24xlarge"                             # Instance Type (default: p4de.24xlarge)
    instance_n_gpus: int = 8                                            # Number of GPUs per Instance

    # OpenVLA Training Parameters
    vla_type: str = (                                                   # Unique VLA ID (specifies config)
        VLARegistry.DINOSIGLIP_224PX_MX_OXE_MAGIC_SOUP_PLUS.vla_id
    )

    # Updated Paths for Data / Runs (on Sagemaker Volume)
    data_root_dir: str = (                                              # Open-X Data Root (in Sagemaker Volume)
        "/opt/ml/input/data/training/surajnair/datasets/openx_processed"
    )
    run_root_dir: str = (                                               # Run/Logs Root (in Sagemaker Volume)
        "/opt/ml/input/data/training/x-openvla/runs"
    )

    # Resume Run Parameters
    pretrained_checkpoint: Optional[str] = (
        "/opt/ml/input/data/training/x-openvla/runs/prism-dinosiglip-224px+mx-oxe-magic-soup-plus+n8+b32+x7/"
        "checkpoints/step-187500-epoch-25-loss=0.6089.pt"
    )
    resume_step: Optional[int] = 187500
    resume_epoch: Optional[int] = 25

    # Sagemaker Job Parameters
    entry_point: str = "vla-scripts/train.py"                           # Entry Point for Training
    input_source: str = "lustre"                                        # Data source in < lustre >
    image_uri: str = (                                                  # Path to Sagemaker Docker Image (in AWS ECR)
        "124224456861.dkr.ecr.us-east-1.amazonaws.com/openvla:latest"
    )
    max_days: int = 10                                                  # Cutoff for Training Time

    # Weights & Biases API Key
    wandb_api_key: Union[str, Path] = Path(".wandb_api_key")            # W&B API Key (for real-time logging)

    # Local Debugging
    debug: bool = False                                                 # Launch Sagemaker Debugging (on `localhost`)
    # fmt: on


@draccus.wrap()
def launch(cfg: LaunchConfig) -> None:
    print("[*] Configuring Sagemaker Launch =>> OpenVLA Training")

    # Parse & Verify W&B API Key
    print("[*] Verifying W&B API Key")
    wandb_api_key = cfg.wandb_api_key.read_text().strip() if isinstance(cfg.wandb_api_key, Path) else cfg.wandb_api_key
    assert wandb.login(key=wandb_api_key, verify=True), "Invalid W&B API Key!"

    # Initialize Sagemaker Session
    print(f"[*] Initializing Sagemaker Session\n\t=>> Role ARN: `{ROLE_ARN}`")
    sagemaker_session = sagemaker.Session() if not cfg.debug else sagemaker.LocalSession()

    # Assemble Job Hyperparameters
    #   =>> Note: For future `S3` support, make sure to set `input_mode = "FastFile"` in Pytorch Estimator init
    print(f"[*] Assembling Job Parameters =>> VLA Type: `{cfg.vla_type}`")
    assert cfg.input_source == "lustre", f"Found `{cfg.input_source = }`; we currently only support `lustre`!"
    train_fs = FileSystemInput(**LUSTRE_PARAMETERS)
    hyperparameters = {
        "vla.type": cfg.vla_type,
        "data_root_dir": cfg.data_root_dir,
        "run_root_dir": cfg.run_root_dir,
        "pretrained_checkpoint": cfg.pretrained_checkpoint,
        "resume_step": cfg.resume_step,
        "resume_epoch": cfg.resume_epoch,
    }

    # Launch!
    print("[*] Creating Sagemaker Estimator =>> Launching!")
    estimator = PyTorch(
        role=ROLE_ARN,
        base_job_name=cfg.job_name,
        instance_count=cfg.instance_count,
        instance_type=cfg.instance_type if not cfg.debug else "local_gpu",
        entry_point=cfg.entry_point,
        image_uri=cfg.image_uri,
        hyperparameters=hyperparameters,
        environment={
            "PYTHONPATH": "/opt/ml/code",
            "WANDB_API_KEY": wandb_api_key,
            "HF_HOME": "/opt/ml/input/data/training/skaramcheti/cache",
            "TF_CPP_MIN_LOG_LEVEL": "3",
        },
        sagemaker_session=sagemaker_session,
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUP_IDS,
        keep_alive_period_in_seconds=3600,
        max_run=60 * 60 * 24 * cfg.max_days,
        distribution={"torch_distributed": {"enabled": True}},
        disable_profiler=True,
        tags=[
            {"Key": "tri.project", "Value": "LBM:PJ-0109"},
            {"Key": "tri.owner.email", "Value": "siddharth.karamcheti@tri.global"},
        ],
    )
    estimator.fit(inputs={"training": train_fs if not cfg.debug else "file:///mnt/fsx/"})


if __name__ == "__main__":
    launch()
