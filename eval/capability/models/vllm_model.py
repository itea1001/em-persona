import torch
from vllm import LLM, SamplingParams

from .base import BaseModel


class VLLMModel(BaseModel):
    def __init__(
        self,
        model: str,
        tensor_parallel_size: int | None = None,
        gpu_memory_utilization: float = 0.95,
        max_model_len: int | None = None,
        **kwargs,
    ):
        if tensor_parallel_size is None:
            tensor_parallel_size = torch.cuda.device_count()
        self.model_path = model
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enable_prefix_caching=True,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate(
        self,
        messages: list[list[dict]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> list[str]:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

        texts = [
            self.tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=True
            )
            for m in messages
        ]

        completions = self.llm.generate(texts, sampling_params, use_tqdm=True)
        return [c.outputs[0].text for c in completions]
