"""Stage 2: text-to-image backends. Each backend renders a caption-free cartoon scene."""
import torch

STYLE = (
    "Bold cartoon comic illustration, thick clean black outlines, flat vibrant colors, "
    "cel shading, exaggerated funny facial expressions, simple uncluttered composition, "
    "plain solid bright background, the main characters centered with empty space along "
    "the top and bottom edges. Absolutely no text, no letters, no words, no captions, no speech bubbles."
)

NO_TEXT = "Absolutely no text, no letters, no words, no captions, no speech bubbles, no signs with writing."

MODELS = {
    # Apache-2.0, 6B DiT, 8-step distilled. Commercial use OK. Default.
    "zimage": "Tongyi-MAI/Z-Image-Turbo",
    # Qwen Research License (non-commercial), 7B DiT, 8-step distilled.
    "qwen": "Qwen/Qwen-Image-2.1-Turbo",
}


class ImageBackend:
    def __init__(self, name: str = "zimage", device: str = "cuda:0"):
        self.name = name
        if name == "zimage":
            from diffusers import ZImagePipeline
            self.pipe = ZImagePipeline.from_pretrained(MODELS[name], torch_dtype=torch.bfloat16)
        elif name == "qwen":
            from diffusers import QwenImage21Pipeline
            self.pipe = QwenImage21Pipeline.from_pretrained(MODELS[name], torch_dtype=torch.bfloat16)
        else:
            raise ValueError(f"unknown model {name!r}; choose from {list(MODELS)}")
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)
        self.device = device

    @torch.inference_mode()
    def __call__(self, scene: str, width: int, height: int, seed: int, style: str = STYLE):
        prompt = f"{scene.strip()} {style.strip()}"
        if "no text" not in prompt.lower():
            prompt += " " + NO_TEXT
        g = torch.Generator("cpu").manual_seed(seed)
        if self.name == "zimage":
            return self.pipe(prompt=prompt, width=width, height=height, num_inference_steps=9,
                             guidance_scale=0.0, generator=g).images[0]
        return self.pipe(prompt=prompt, width=width, height=height, use_kv_cache=True,
                         generator=g).images[0]
