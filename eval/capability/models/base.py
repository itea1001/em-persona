from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def generate(
        self,
        messages: list[list[dict]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> list[str]:
        """Generate responses for a batch of message lists. Returns list of strings."""
        pass


def load_model(model_name_or_path: str, **kwargs) -> BaseModel:
    """Factory: auto-detect vLLM local vs API model."""
    if any(model_name_or_path.startswith(p) for p in ("gpt-", "o1", "o3", "o4")):
        from .openai_model import OpenAIModel
        return OpenAIModel(model=model_name_or_path, **kwargs)
    elif model_name_or_path.startswith("claude-"):
        from .anthropic_model import AnthropicModel
        return AnthropicModel(model=model_name_or_path, **kwargs)
    else:
        from .vllm_model import VLLMModel
        return VLLMModel(model=model_name_or_path, **kwargs)
