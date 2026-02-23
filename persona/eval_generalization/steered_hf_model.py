"""HuggingFace model with activation steering, implementing BaseModel interface.

Wraps a HuggingFace CausalLM with an ActivationSteerer forward hook so that
it can be plugged into the alignment eval framework (original_em, strongreject, etc.).
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root so we can import eval.capability.models
sys.path.insert(0, "/net/scratch2/mingxuanl/em-persona")
from eval.capability.models.base import BaseModel

# Copy ActivationSteerer inline to keep em-persona self-contained
class ActivationSteerer:
    """Add (coeff * steering_vector) to a chosen transformer block's output."""

    _POSSIBLE_LAYER_ATTRS = (
        "transformer.h", "encoder.layer", "model.layers",
        "gpt_neox.layers", "block",
    )

    def __init__(self, model, steering_vector, *, coeff=1.0, layer_idx=-1):
        self.model = model
        self.coeff = float(coeff)
        self.layer_idx = layer_idx
        self._handle = None

        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1-D")

    def _locate_layer(self):
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            for part in path.split("."):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    break
            else:
                if hasattr(cur, "__getitem__"):
                    if -len(cur) <= self.layer_idx < len(cur):
                        return cur[self.layer_idx]
        raise ValueError("Could not find layer list on the model.")

    def _hook_fn(self, module, ins, out):
        steer = self.coeff * self.vector

        def _add(t):
            return t + steer.to(t.device)

        if torch.is_tensor(out):
            return _add(out)
        elif isinstance(out, (tuple, list)):
            if torch.is_tensor(out[0]):
                return (_add(out[0]), *out[1:])
        return out

    def __enter__(self):
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        if self._handle:
            self._handle.remove()
            self._handle = None


class SteeredHFModel(BaseModel):
    """HuggingFace model with optional activation steering.

    When persona_vector and layer/coeff are set, adds the steering hook
    during generation. When coeff=0 or no vector, runs plain HF generation.
    """

    def __init__(
        self,
        model_path: str,
        persona_vector_path: str | None = None,
        layer_idx: int = 15,
        coeff: float = 0.0,
        device: str = "cuda:0",
    ):
        self.device = device
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map=device, torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

        self.coeff = coeff
        self.layer_idx = layer_idx
        self.steering_vector = None

        if persona_vector_path and coeff != 0.0:
            print(f"Loading persona vector from {persona_vector_path}...")
            vec = torch.load(persona_vector_path, map_location="cpu")
            # vec shape: [num_layers, hidden_dim] — pick the target layer
            if vec.ndim == 2:
                self.steering_vector = vec[layer_idx].float()
            else:
                self.steering_vector = vec.float()
            print(f"  Layer {layer_idx}, coeff={coeff}, vec norm={self.steering_vector.norm():.4f}")

    def generate(
        self,
        messages: list[list[dict]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> list[str]:
        """Generate responses, one at a time (HF doesn't batch well with hooks)."""
        results = []
        total = len(messages)

        # Set up steering context
        use_steering = self.steering_vector is not None and self.coeff != 0.0

        for i, msgs in enumerate(messages):
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  Generating {i+1}/{total}...")

            text = self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = dict(
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            if temperature > 0:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 1.0

            with torch.no_grad():
                if use_steering:
                    with ActivationSteerer(
                        self.model,
                        self.steering_vector,
                        coeff=self.coeff,
                        layer_idx=self.layer_idx,
                    ):
                        output_ids = self.model.generate(**inputs, **gen_kwargs)
                else:
                    output_ids = self.model.generate(**inputs, **gen_kwargs)

            new_tokens = output_ids[0, input_len:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(response)

        return results
