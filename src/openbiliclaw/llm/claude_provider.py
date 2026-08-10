"""Anthropic Claude LLM provider."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

import httpx
from anthropic import AsyncAnthropic

from .base import (
    DEFAULT_REASONING_EFFORT,
    LLM_USER_AGENT,
    LLMAuthError,
    LLMProvider,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMResponseError,
    LLMTimeoutError,
)

if TYPE_CHECKING:
    from anthropic.types import Message, MessageParam

logger = logging.getLogger(__name__)


class ClaudeProvider(LLMProvider):
    """Anthropic Claude provider."""

    _MAX_RETRIES = 3
    _BASE_RETRY_DELAY = 0.25

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = 1200.0,
        base_url: str = "",
        proxy: str = "",
        trust_env: bool = True,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> None:
        self._model = model
        self._reasoning_effort = reasoning_effort.strip()
        self.base_url = base_url.strip()
        # Overseas routing policy mirrors OpenAIProvider.
        self._proxy = proxy.strip()
        self._trust_env = bool(trust_env and not self._proxy)
        client_kwargs: dict[str, Any] = {}
        if self._proxy or not self._trust_env:
            httpx_kwargs: dict[str, Any] = {"timeout": timeout, "trust_env": self._trust_env}
            if self._proxy:
                httpx_kwargs["proxy"] = self._proxy
            client_kwargs["http_client"] = httpx.AsyncClient(**httpx_kwargs)
        # Empty base_url → SDK default (https://api.anthropic.com). A custom
        # value points at any Anthropic-protocol gateway serving /v1/messages
        # (third-party relays, LiteLLM, etc.) — see issue #72.
        self._client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
            base_url=self.base_url or None,
            # Override the SDK's default ``AsyncAnthropic/Python`` UA, which
            # CF-fronted relays block with HTTP 403. ``default_headers`` is
            # spread last by the SDK, so this wins over the built-in UA.
            default_headers={"User-Agent": LLM_USER_AGENT},
            **client_kwargs,
        )

    @property
    def name(self) -> str:
        return "claude"

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        reasoning_effort: str | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        effective_model = (model or "").strip() or self._model
        # Extract system message if present
        system = ""
        chat_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                chat_messages.append(msg)

        # v0.3.29+: Anthropic prompt-cache integration. Claude requires
        # explicit ``cache_control: {"type": "ephemeral"}`` markers on
        # the message blocks we want cached — pure plain-string
        # ``system="..."`` is never cached. We always mark the system
        # block as cacheable; Anthropic silently ignores the marker if
        # the system text is below the per-model min (1024 tok Sonnet,
        # 2048 tok Haiku/Opus), so this is safe for short prompts too.
        # Cache hit gets billed at 10% of input rate; the first call
        # writes cache at +25% surcharge, then 5min TTL on reads. The
        # system_param goes through ``_render_system_param`` which the
        # tests can override.
        system_text = system or "You are a helpful assistant."
        system_param: Any = self._render_system_param(system_text)

        request_kwargs: dict[str, Any] = {
            "model": effective_model,
            "max_tokens": max_tokens,
            "system": system_param,
            "messages": chat_messages,
            "temperature": temperature,
        }
        effort = self._claude_effort(effective_model, reasoning_effort)
        if effort is not None:
            request_kwargs["output_config"] = {"effort": effort}

        response = cast(
            "Message",
            await self._request_with_retry(**request_kwargs),
        )

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        if not content.strip():
            raise LLMResponseError("claude returned empty content")

        # Claude exposes cache fields when prompt-cache is in use:
        # cache_read_input_tokens (90% off) + cache_creation_input_tokens
        # (+25% surcharge). We surface them under the universal
        # ``cached_input_tokens`` / ``cache_creation_input_tokens`` keys
        # so downstream pricing / observability is provider-agnostic.
        cache_read = int(getattr(response.usage, "cache_read_input_tokens", 0) or 0)
        cache_create = int(getattr(response.usage, "cache_creation_input_tokens", 0) or 0)
        usage_dict = {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }
        if cache_read:
            usage_dict["cached_input_tokens"] = cache_read
        if cache_create:
            usage_dict["cache_creation_input_tokens"] = cache_create
        return LLMResponse(
            content=content,
            model=response.model,
            provider="claude",
            usage=usage_dict,
            raw=response,
        )

    def _render_system_param(self, system_text: str) -> Any:
        """Wrap the system prompt in Anthropic's prompt-cache shape.

        The Claude API accepts ``system`` as either a plain string or a
        list of typed blocks; only the latter form supports
        ``cache_control``. We always emit the list form with an
        ``ephemeral`` cache marker on the system block. If the system
        text is below the per-model minimum (1024 tok Sonnet / 2048
        tok Haiku/Opus), Anthropic silently ignores the marker rather
        than erroring, so this is safe regardless of size.
        """
        return [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def _claude_effort(self, model: str, requested: str | None) -> str | None:
        """Map portable effort onto Claude models that expose output_config."""

        if not self._supports_effort(model):
            return None
        effort = self._reasoning_effort if requested is None else requested.strip().lower()
        if effort in {"", "none", "minimal"}:
            return "low"
        # ``xhigh`` is not present in every installed Anthropic SDK/model and
        # ``max`` is model-specific.  High is the safe portable upper level.
        if effort in {"max", "xhigh"}:
            return "high"
        if effort in {"low", "medium", "high"}:
            return effort
        return DEFAULT_REASONING_EFFORT

    @staticmethod
    def _supports_effort(model: str) -> bool:
        name = model.strip().lower().replace(".", "-")
        return any(
            marker in name
            for marker in (
                "claude-fable-5",
                "claude-mythos-5",
                "claude-mythos-preview",
                "claude-opus-4-5",
                "claude-opus-4-6",
                "claude-opus-4-7",
                "claude-opus-4-8",
                "claude-sonnet-4-6",
                "claude-sonnet-5",
            )
        )

    async def _request_with_retry(self, **kwargs: Any) -> Any:
        """Send a request with bounded retry for transient failures."""
        last_error: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                request_kwargs: dict[str, Any] = {
                    "model": cast("str", kwargs["model"]),
                    "max_tokens": cast("int", kwargs["max_tokens"]),
                    "system": kwargs["system"],
                    "messages": cast("list[MessageParam]", kwargs["messages"]),
                    "temperature": cast("float", kwargs["temperature"]),
                }
                if "output_config" in kwargs:
                    request_kwargs["output_config"] = kwargs["output_config"]
                return await self._client.messages.create(**request_kwargs)
            except Exception as exc:
                mapped = self._map_error(exc)
                last_error = mapped
                if not self._is_retryable(mapped) or attempt == self._MAX_RETRIES:
                    raise mapped from exc

                await asyncio.sleep(self._BASE_RETRY_DELAY * attempt)

        if last_error is None:
            raise LLMProviderError("claude request failed")
        raise last_error

    def _map_error(self, exc: Exception) -> LLMProviderError:
        """Map Anthropic or network errors into shared provider errors."""
        if isinstance(exc, LLMProviderError):
            return exc
        if isinstance(exc, TimeoutError):
            return LLMTimeoutError("claude request timed out")

        message = str(exc).lower()
        if "rate limit" in message or "too many requests" in message:
            return LLMRateLimitError("claude rate limit exceeded")
        if getattr(exc, "status_code", None) == 401:
            logger.warning("claude rejected our credentials with HTTP 401: %s", exc)
            return LLMAuthError(
                f"claude authentication failed: HTTP 401: {exc}",
                provider_name="claude",
                endpoint=self.base_url,
            )

        return LLMProviderError(f"claude request failed: {exc}")

    def _is_retryable(self, exc: LLMProviderError) -> bool:
        """Whether a mapped exception should be retried."""
        # See OpenAIProvider._is_retryable: a 401 is terminal until the user
        # fixes their key, so retrying only delays the actionable error.
        if isinstance(exc, (LLMRateLimitError, LLMAuthError)):
            return False
        return isinstance(exc, (LLMProviderError, LLMTimeoutError))
