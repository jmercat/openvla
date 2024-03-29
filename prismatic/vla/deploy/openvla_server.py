# On action server: pip install uvicorn fastapi json-numpy
# On client: pip install requests json-numpy

"""
On client (for server running on 0.0.0.0:8000):
    import requests
    import json_numpy
    json_numpy.patch()

    action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": np.zeros((256, 256, 3), dtype=np.uint8), "instruction": "do something"}
    ).json()

If your server is not reachable from the open internet, you can forward ports to your client via ssh:
    ssh -L 8000:128.32.162.191:8000 ssh karl@128.32.162.191
"""


import json_numpy

json_numpy.patch()

import logging  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Union  # noqa: E402

import draccus  # noqa: E402
import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from PIL import Image  # noqa: E402

from prismatic.models.load import load_vla  # noqa: E402
from prismatic.models.vlms import OpenVLA  # noqa: E402


@dataclass
class VLAServerConfig:
    # fmt: off
    # Pre-trained VLA model checkpoint to serve
    checkpoint_path: Union[str, Path] = Path(
        "/shared/karl/models/open_vla/lr-2e5+siglip-224px+mx-bridge+n1+b32+x7/step-080000-epoch-09-loss=0.0987.pt"
    )

    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # HF Hub Credentials (for LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")  # Environment variable or Path to HF Token
    # fmt: on


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))


class OpenVLAServer:
    """A simple server for OpenVLA policies.

    Use /act to predict an action for a given image + instruction.
        - Takes in {'image': np.ndarray, 'instruction': str}
        - Returns {'action': np.ndarray}
    """

    def __init__(self, checkpoint_path, hf_token=None):
        self.vla: OpenVLA = load_vla(checkpoint_path, hf_token=hf_token)

    def run(self, host="0.0.0.0", port=8000):
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

    def predict_action(self, payload: Dict[Any, Any]):
        try:
            image = payload["image"]
            instruction = payload["instruction"]
            action = self.vla.predict_action(Image.fromarray(image).convert("RGB"), instruction)
            return JSONResponse(action)
        except:  # noqa: E722        # blanket except to robustify against external errors
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error. "
                "Make sure your request complies with the expected format: \n"
                "{'image': np.ndarray, 'instruction': str}"
            )
            return "error"


@draccus.wrap()
def run_vla_server(cfg: VLAServerConfig):
    server = OpenVLAServer(cfg.checkpoint_path, cfg.hf_token)
    server.run(cfg.host, cfg.port)


if __name__ == "__main__":
    run_vla_server()
