import asyncio
import os
import logging

import openai
import tqdm
from openai import AsyncOpenAI, OpenAI

from .base import BaseModel
from .rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class OpenAIModel(BaseModel):
    EXCEPTIONS_TO_CATCH = (
        openai.RateLimitError,
        openai.APIError,
        openai.APITimeoutError,
    )

    def __init__(
        self,
        model: str,
        max_retry: int = 300,
        min_backoff: float = 1.0,
        max_backoff: float = 20.0,
        timeout: int = 120,
        max_concurrent: int = 16,
        **kwargs,
    ):
        self.model = model
        self.max_retry = max_retry
        self.timeout = timeout
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter(min_backoff=min_backoff, max_backoff=max_backoff)
        self.api = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        self.rate_limiter.add_event()
        for _ in range(self.max_retry):
            try:
                resp = self.api.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=self.timeout,
                    **kwargs,
                )
                return resp.choices[0].message.content
            except self.EXCEPTIONS_TO_CATCH as e:
                self.rate_limiter.backoff(e)
                continue
        raise RuntimeError(
            f"Max retry ({self.max_retry}) exceeded for OpenAI API request."
        )

    def _batched_generate(
        self,
        messages: list[list[dict]],
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> list[str]:
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        status_bar = tqdm.tqdm(total=len(messages), desc="OpenAI API")

        async def _async_generate(sem, msgs):
            async with sem:
                for attempt in range(self.max_retry):
                    try:
                        resp = await client.chat.completions.create(
                            messages=msgs,
                            model=self.model,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout=self.timeout,
                            **kwargs,
                        )
                        status_bar.update(1)
                        self.rate_limiter.add_event()
                        return resp.choices[0].message.content
                    except self.EXCEPTIONS_TO_CATCH as e:
                        self.rate_limiter.backoff(e)
                        error_str = str(e).lower()
                        if any(
                            s in error_str
                            for s in ("content management policy", "contentfilter", "content filter")
                        ):
                            if attempt >= 10:
                                logger.warning(f"Content filter error after {attempt+1} retries: {e}")
                                return ""
                        continue
                logger.warning(f"Max retry ({self.max_retry}) exceeded. Returning empty string.")
                return ""

        sem = asyncio.Semaphore(self.max_concurrent)
        tasks = [_async_generate(sem, msgs) for msgs in messages]
        results = asyncio.run(asyncio.gather(*tasks))
        status_bar.close()

        none_count = sum(1 for r in results if r == "")
        if none_count > 0:
            logger.warning(f"{none_count}/{len(results)} requests returned empty.")

        return results
