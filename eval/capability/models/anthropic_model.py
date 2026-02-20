import asyncio
import copy
import os
import logging

import anthropic
import tqdm
from anthropic import AsyncAnthropic, Anthropic

from .base import BaseModel
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class AnthropicModel(BaseModel):
    EXCEPTIONS_TO_CATCH = (
        anthropic.RateLimitError,
        anthropic.APIError,
        anthropic.APITimeoutError,
    )

    def __init__(
        self,
        model: str,
        max_retry: int = 30,
        min_backoff: float = 1.0,
        max_backoff: float = 60.0,
        max_concurrent: int = 3,
        **kwargs,
    ):
        self.model = model
        self.max_retry = max_retry
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(min_backoff=min_backoff, max_backoff=max_backoff)
        self.api = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    @staticmethod
    def _extract_system(messages: list[dict]) -> tuple[str, list[dict]]:
        """Extract system message without mutating the input list."""
        msgs = copy.deepcopy(messages)
        system_prompt = ""
        for idx, msg in enumerate(msgs):
            if msg["role"] == "system":
                system_prompt = msgs.pop(idx)["content"]
                break
        return system_prompt, msgs

    def generate(
        self,
        messages: list[list[dict]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ) -> list[str]:
        if len(messages) == 0:
            return []
        if len(messages) == 1:
            return [self._generate_single(messages[0], temperature, max_tokens, **kwargs)]
        return self._batched_generate(messages, temperature, max_tokens, **kwargs)

    def _generate_single(
        self,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> str:
        system_prompt, msgs = self._extract_system(messages)
        for _ in range(self.max_retry):
            try:
                resp = self.api.messages.create(
                    system=system_prompt,
                    messages=msgs,
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return resp.content[0].text
            except self.EXCEPTIONS_TO_CATCH as e:
                self.rate_limiter.backoff(e)
                continue
            except anthropic.BadRequestError as e:
                logger.warning(f"Content filtered: {e}")
                return ""
        raise RuntimeError(
            f"Max retry ({self.max_retry}) exceeded for Anthropic API request."
        )

    def _batched_generate(
        self,
        messages: list[list[dict]],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> list[str]:
        client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        status_bar = tqdm.tqdm(total=len(messages), desc="Anthropic API")

        async def _async_generate(sem, raw_msgs):
            system_prompt, msgs = self._extract_system(raw_msgs)
            async with sem:
                for _ in range(self.max_retry):
                    try:
                        resp = await client.messages.create(
                            system=system_prompt,
                            messages=msgs,
                            model=self.model,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            **kwargs,
                        )
                        status_bar.update(1)
                        self.rate_limiter.add_event()
                        return resp.content[0].text
                    except self.EXCEPTIONS_TO_CATCH as e:
                        self.rate_limiter.backoff(e)
                        continue
                    except anthropic.BadRequestError:
                        logger.warning("Content filtered, returning empty string.")
                        return ""
                logger.warning(f"Max retry ({self.max_retry}) exceeded. Returning empty string.")
                return ""

        sem = asyncio.Semaphore(self.max_concurrent)
        tasks = [_async_generate(sem, msgs) for msgs in messages]
        results = asyncio.run(asyncio.gather(*tasks))
        status_bar.close()
        return results
