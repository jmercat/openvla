import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

VLA_PATH = "openvla/openvla-7b-v01"

SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"


device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
vla = AutoModelForVision2Seq.from_pretrained(
    VLA_PATH,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
    # load_in_4bit=True
).to(device)
processor = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)

total_time = 0.0
for _ in range(100):
    image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
    instruction = "do something"
    unnorm_key = "bridge_orig"
    start = time.time()
    prompt = get_openvla_prompt(instruction)
    inputs = processor(prompt, image.convert("RGB")).to(device, dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    run_time = time.time() - start
    total_time += run_time
    print(run_time)

print(f"\nAverage runtime: {total_time / 100}")
