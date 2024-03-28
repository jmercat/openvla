from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from prismatic.models.vlms.prismatic import PrismaticVLM
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class OpenVLA(PrismaticVLM):
    """
    OpenVLA model class.
    Usage:
        from prismatic.models.load import load_vla
        vla = load_vla(model_path)
        action = vla.predict_action(
            image,
            instruction,
        )       # predicts unnormalized, continuous action

    """
    def __init__(
        self,
        action_norm_stats: Dict[str, Dict[str, List[float]]],
        action_tokenizer: ActionTokenizer,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.action_norm_stats = action_norm_stats
        self.action_tokenizer = action_tokenizer

    @torch.inference_mode()
    def predict_action(self, image: Image, instruction: str, **kwargs: str) -> np.ndarray:
        # For now, only support generation with a batch size of 1 for simplicity
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # Build VLA prompt
        prompt_builder = self.get_prompt_builder()
        prompt_builder.add_turn(
            role="human",
            message=f"What action should the robot take to {instruction.lower()}?"
        )
        prompt_text = prompt_builder.get_prompt()

        # Prepare Inputs
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)

        # TODO (karl): figure out how to make this tokenizer-independent
        # Note (Moo Jin): We need to add this special empty token ('') after the colon (':') token in "ASSISTANT:"
        # in order for the predictions to match the training configuration and be accurate.
        input_ids = torch.cat(
            (
                input_ids,
                torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(self.device)
            ), dim=1
        )

        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super(PrismaticVLM, self).generate(
                input_ids=input_ids,  # Shape: [1, seq]
                pixel_values=pixel_values,  # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs
            )
            # fmt: on

        # Extract predicted action tokens and translate into (normalized) continuous actions
        predicted_action_token_ids = generated_ids[:, -self.action_dim:]
        normalized_actions = self.action_tokenizer.decode_token_ids_to_actions(
            predicted_action_token_ids.cpu().numpy()
        )

        # Unnormalize actions
        mask = self.action_norm_stats.get(
            "mask", np.ones_like(self.action_norm_stats["mean"], dtype=bool)
        )
        action_high, action_low = np.array(self.action_norm_stats["q99"]), np.array(self.action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @property
    def action_dim(self):
        """Dimensionality of the policy's action space."""
        return len(self.action_norm_stats["action"]["q01"])