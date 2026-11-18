"""Thin batched wrapper around a local Qwen3.5 (vision-language) model.

One set of weights serves several agents (writer, art director, critic, grounder);
each agent is just a different system prompt. Pass a different model id per agent
in memeloop.py if you want truly separate models.
"""
import json
import re

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


def extract_json(text: str):
    """Return the first JSON object/array found in text (handles ```json fences / <think>)."""
    text = text.split("</think>")[-1]
    dec = json.JSONDecoder()
    for m in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
            return obj
        except json.JSONDecodeError:
            continue
    return None


def msg(role, *parts):
    """msg("user", "text", pil_image, "more text") -> chat message with typed content."""
    content = [{"type": "image", "image": p} if not isinstance(p, str) else {"type": "text", "text": p}
               for p in parts]
    return {"role": role, "content": content}


class LLM:
    def __init__(self, model_id="Qwen/Qwen3.5-9B", device="cuda:0"):
        self.device = device
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device).eval()

    @torch.inference_mode()
    def chat(self, convs, max_new_tokens=2048, temperature=0.7, think=False):
        enc = self.proc.apply_chat_template(
            convs, add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", padding=True, enable_thinking=think).to(self.device)
        kw = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False)
        out = self.model.generate(**enc, max_new_tokens=max_new_tokens, **kw)
        return self.proc.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def chat_json(self, convs, validate=lambda x: x, retries=3, batch_size=10, **kw):
        """Batched chat returning validate(parsed_json) per conversation; None if it kept failing."""
        results = [None] * len(convs)
        for _ in range(retries):
            todo = [i for i, r in enumerate(results) if r is None]
            for s in range(0, len(todo), batch_size):
                idx = todo[s:s + batch_size]
                for i, text in zip(idx, self.chat([convs[i] for i in idx], **kw)):
                    obj = extract_json(text)
                    try:
                        results[i] = validate(obj) if obj is not None else None
                    except (KeyError, TypeError, ValueError, AttributeError):
                        results[i] = None
            if all(r is not None for r in results):
                break
        return results
