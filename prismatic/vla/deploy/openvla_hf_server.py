# On action server: pip install uvicorn fastapi json-numpy
# On client: pip install requests json-numpy

"""
On client (for server running on 0.0.0.0:8000):

import requests
import json_numpy
json_numpy.patch()
import numpy as np

action = requests.post(
    "http://0.0.0.0:8000/act",
    json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
).json()

If your server is not reachable from the open internet, you can forward ports to your client via ssh:
    ssh -L 8000:localhost:8000 ssh karl@128.32.162.191
"""


import json_numpy

json_numpy.patch()

import json  # noqa: E402
import logging  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Union  # noqa: E402

import draccus  # noqa: E402
import torch  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoModelForVision2Seq, AutoProcessor  # noqa: E402

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass
class VLAServerConfig:
    # fmt: off
    # Pre-trained VLA model checkpoint to serve
    checkpoint_path: Union[str, Path] = Path(
        "/raid/users/karl/models/openvla-7b-v01_droid_wipe_bs16_lr2e-05_lora_r32_dropout0.0"
    )

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")  # Environment variable or Path to HF Token
    # fmt: on


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))


def get_openvla_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"


class OpenVLAServer:
    """A simple server for OpenVLA policies.

    Use /act to predict an action for a given image + instruction.
        - Takes in {'image': np.ndarray, 'instruction': str, 'unnorm_key': optional(str)}
        - Returns {'action': np.ndarray}
    """

    def __init__(self, checkpoint_path):
        # Load VLA model from HuggingFace
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(self.device)

        # Hack: load dataset stats separately since we forgot to update during finetuning
        with open(checkpoint_path / "dataset_statistics.json", "r") as f:
            norm_stats = json.load(f)
        self.vla.norm_stats = norm_stats

    def run(self, host="0.0.0.0", port=8000):
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

    def predict_action(self, payload: Dict[Any, Any]):
        try:
            if double_encode := "encoded" in payload:
                # This shim supports server evals at Google, where json_numpy is hard to install
                # So need to "double-encode" numpy arrays to send them as strings.
                assert len(payload.keys()) == 1, "Only uses encoded payload."
                payload = json.loads(payload["encoded"])
            image = payload["image"]
            instruction = payload["instruction"]
            unnorm_key = payload.get("unnorm_key")

            # Run VLA inference
            prompt = get_openvla_prompt(instruction)
            inputs = self.processor(prompt, Image.fromarray(image).convert("RGB")).to(self.device, dtype=torch.bfloat16)
            action = self.vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            if double_encode:
                return JSONResponse(json_numpy.dumps(action))
            else:
                return JSONResponse(action)
        except:  # noqa: E722        # blanket except to robustify against external errors
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error. "
                "Make sure your request complies with the expected format: \n"
                "{'image': np.ndarray, 'instruction': str} \n"
                "You can optionally add an `unnorm_key: str` to specify the dataset stats you want to use "
                "for un-normalizing the output actions."
            )
            return "error"


@draccus.wrap()
def run_vla_server(cfg: VLAServerConfig):
    server = OpenVLAServer(cfg.checkpoint_path)
    server.run(cfg.host, cfg.port)


if __name__ == "__main__":
    run_vla_server()
