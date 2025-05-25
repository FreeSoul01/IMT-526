# image_gen.py

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import os

class ImageGenerator:
    def __init__(self, model_id="CompVis/stable-diffusion-v1-4", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def generate(self, prompt, seed=None, num_steps=50, width=512, height=512, out_path=None):
        generator = torch.Generator(self.device).manual_seed(seed) if seed else None
        image = self.pipe(prompt, num_inference_steps=num_steps, height=height, width=width, generator=generator).images[0]
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            image.save(out_path)
        return image
