"""Gemini Developer API provider built on the official google-genai SDK."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, NoReturn

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

logger = logging.getLogger(__name__)

genai: Any | None
errors: Any | None
types: Any | None
_SDK_IMPORT_ERROR: str | None

# Gemini 2.5 exposes token budgets rather than effort names.  These ratios
# follow OpenRouter's documented cross-provider effort mapping (10/20/50/80/95
# percent) so ``medium`` remains a balanced default while leaving half of the
# output window for the final answer.  Model-specific limits are applied below.
_GEMINI_25_EFFORT_RATIOS = {
    "minimal": 0.10,
    "low": 0.20,
    "medium": 0.50,
    "high": 0.80,
    "xhigh": 0.95,
    "max": 0.95,
}

try:
    from google import genai as _genai
    from google.genai import errors as _errors
    from google.genai import types as _types
except ImportError as _exc:  # pragma: no cover - exercised via subprocess regression test
    # ImportError (not just ModuleNotFoundError): the SDK may be installed yet
    # fail to load when a native transitive dep breaks — e.g. cryptography's
    # manylinux wheel can't dlopen under Termux/Android Bionic (issue #80).
    # Either way the provider must degrade instead of crashing CLI startup.
    genai = None
    errors = None
    types = None
    _SDK_IMPORT_ERROR = str(_exc)
else:
    genai = _genai
    errors = _errors
    types = _types
    _SDK_IMPORT_ERROR = None


def gemini_sdk_available() -> bool:
    """Return whether the optional google-genai dependency is installed and loadable."""
    return genai is not None and types is not None


def _raise_missing_sdk() -> NoReturn:
    detail = f" (import failed: {_SDK_IMPORT_ERROR})" if _SDK_IMPORT_ERROR else ""
    raise LLMProviderError(
        "Gemini provider requires the optional dependency 'google-genai' to be "
        f"installed and loadable.{detail}"
    )


class GeminiProvider(LLMProvider):
    """Gemini provider using the official Gemini Developer API client."""

    supports_embedding = True
    # Class can implement image embed; actual readiness depends on the
    # embedding model name (gemini-embedding-2 family only).
    supports_image_embedding = True

    _MAX_RETRIES = 3
    _BASE_RETRY_DELAY = 0.25
    _MULTIMODAL_EMBEDDING_MARKERS = ("gemini-embedding-2",)

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        timeout: float = 1200.0,
        base_url: str = "",
        embedding_output_dimensionality: int | None = None,
        proxy: str = "",
        trust_env: bool = True,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    ) -> None:
        if not gemini_sdk_available():
            _raise_missing_sdk()
        assert genai is not None
        self._model = model
        self._reasoning_effort = reasoning_effort.strip()
        self._embedding_output_dimensionality = (
            embedding_output_dimensionality
            if embedding_output_dimensionality is not None and embedding_output_dimensionality > 0
            else None
        )
        # Override the SDK's default UA — CF-fronted relays block the
        # google-genai default with HTTP 403. HttpOptions.headers is the
        # native "additional HTTP headers" field (google/genai/types.py).
        http_options: dict[str, Any] = {
            "timeout": int(timeout * 1000),
            "headers": {"User-Agent": LLM_USER_AGENT},
        }
        normalized_base_url = (base_url or "").strip()
        if normalized_base_url:
            http_options["base_url"] = normalized_base_url.rstrip("/") + "/"
        # google-genai passes these args to its underlying httpx clients.
        self._proxy = proxy.strip()
        self._trust_env = bool(trust_env and not self._proxy)
        if self._proxy or not self._trust_env:
            transport_args: dict[str, Any] = {"trust_env": self._trust_env}
            if self._proxy:
                transport_args["proxy"] = self._proxy
            http_options["client_args"] = dict(transport_args)
            http_options["async_client_args"] = dict(transport_args)
        self._client = genai.Client(
            api_key=api_key,
            http_options=http_options,
        )

    @property
    def name(self) -> str:
        return "gemini"

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
        if types is None:
            _raise_missing_sdk()
        effective_model = (model or "").strip() or self._model
        effective_effort = (
            self._reasoning_effort if reasoning_effort is None else reasoning_effort.strip()
        )
        thinking_config = self._thinking_config_for_effort(
            effective_model,
            effective_effort,
            max_tokens,
        )
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            response_mime_type="application/json" if json_mode else None,
            thinking_config=thinking_config,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = await self._request_with_retry(
            model=effective_model,
            contents=self._render_messages(messages),
            config=config,
        )

        content = response.text or ""
        if not content.strip():
            raise LLMResponseError("gemini returned empty content")

        usage = None
        if response.usage_metadata is not None:
            usage = {
                "prompt_tokens": response.usage_metadata.prompt_token_count or 0,
                "completion_tokens": response.usage_metadata.candidates_token_count or 0,
                "total_tokens": response.usage_metadata.total_token_count or 0,
            }
            # Gemini exposes cached_content_token_count when a previously
            # uploaded explicit cache (Context Caching API) was used.
            # Normalize under the universal ``cached_input_tokens`` key.
            cached = int(getattr(response.usage_metadata, "cached_content_token_count", 0) or 0)
            if cached:
                usage["cached_input_tokens"] = cached

        return LLMResponse(
            content=content,
            model=response.model_version or effective_model,
            provider="gemini",
            usage=usage,
            raw=response,
        )

    @classmethod
    def _thinking_config_for_effort(
        cls,
        model: str,
        effort: str,
        max_tokens: int,
    ) -> Any | None:
        """Translate the portable effort to Gemini 3 levels or 2.5 budgets."""

        if types is None:
            return None
        name = model.strip().lower()
        normalized = effort.strip().lower()
        if name.startswith("gemini-3"):
            return types.ThinkingConfig(
                thinking_level=cls._gemini_3_thinking_level(name, normalized)
            )
        if not name.startswith("gemini-2.5"):
            return None

        budget = cls._gemini_25_thinking_budget(name, normalized, max_tokens)
        if budget is None:
            return None
        return types.ThinkingConfig(thinking_budget=budget)

    @staticmethod
    def _gemini_3_thinking_level(model: str, effort: str) -> Any:
        assert types is not None
        # Gemini 3.1 Pro cannot fully disable thinking and accepts LOW as its
        # cheapest level.  Gemini 3 Pro only accepts LOW/HIGH; the image
        # Flash-Lite variant accepts MINIMAL/HIGH.  Other Gemini 3 models
        # expose the full MINIMAL/LOW/MEDIUM/HIGH ladder.
        only_low_high = model.startswith("gemini-3-pro")
        only_minimal_high = "flash-lite-image" in model
        if effort in {"", "none"}:
            if only_minimal_high:
                return types.ThinkingLevel.MINIMAL
            if only_low_high or "pro" in model:
                return types.ThinkingLevel.LOW
            return types.ThinkingLevel.MINIMAL
        if effort == "minimal":
            return types.ThinkingLevel.LOW if only_low_high else types.ThinkingLevel.MINIMAL
        if effort == "low":
            return types.ThinkingLevel.MINIMAL if only_minimal_high else types.ThinkingLevel.LOW
        if effort == "medium":
            if only_low_high or only_minimal_high:
                return types.ThinkingLevel.HIGH
            return types.ThinkingLevel.MEDIUM
        if effort in {"high", "xhigh", "max"}:
            return types.ThinkingLevel.HIGH
        return types.ThinkingLevel.MEDIUM

    @staticmethod
    def _gemini_25_thinking_budget(model: str, effort: str, max_tokens: int) -> int | None:
        # 2.5 Pro cannot disable thinking; 128 is its documented minimum.
        # Flash and Flash-Lite accept zero as a true thinking-off request.
        if effort in {"", "none"}:
            if "pro" not in model:
                return 0
            return 128 if max_tokens > 128 else None

        normalized = effort if effort in _GEMINI_25_EFFORT_RATIOS else DEFAULT_REASONING_EFFORT
        ratio = _GEMINI_25_EFFORT_RATIOS[normalized]
        minimum = 128 if "pro" in model else (512 if "flash-lite" in model else 1)
        maximum = 32768 if "pro" in model else 24576
        if max_tokens <= minimum:
            return None
        budget = max(minimum, int(max_tokens * ratio))
        return min(budget, maximum, max_tokens - 1)

    async def _request_with_retry(self, **kwargs: Any) -> Any:
        last_error: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                return await self._client.aio.models.generate_content(**kwargs)
            except Exception as exc:
                mapped = self._map_error(exc)
                last_error = mapped
                if not self._is_retryable(mapped) or attempt == self._MAX_RETRIES:
                    raise mapped from exc
                await asyncio.sleep(self._BASE_RETRY_DELAY * attempt)

        if last_error is None:
            raise LLMProviderError("gemini request failed")
        raise last_error

    def _map_error(self, exc: Exception) -> LLMProviderError:
        if isinstance(exc, LLMProviderError):
            return exc
        if isinstance(exc, TimeoutError):
            return LLMTimeoutError("gemini request timed out")

        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        message = (getattr(exc, "message", None) or str(exc)).lower()
        if status_code == 429 or "rate limit" in message or "resource_exhausted" in message:
            return LLMRateLimitError("gemini rate limit exceeded")
        # Gemini reports a bad key as 401 UNAUTHENTICATED, but also as 400
        # INVALID_ARGUMENT with "API key not valid" — key off both.
        if status_code == 401 or "api key not valid" in message or "unauthenticated" in message:
            logger.warning("gemini rejected our credentials: %s", exc)
            return LLMAuthError(
                f"gemini authentication failed: HTTP {status_code or 401}: {exc}",
                provider_name="gemini",
                endpoint="",
            )
        if (errors is not None and isinstance(exc, errors.ServerError)) or (
            status_code and int(status_code) >= 500
        ):
            return LLMProviderError(f"gemini server error: {status_code}")
        return LLMProviderError(f"gemini request failed: {exc}")

    def _is_retryable(self, exc: LLMProviderError) -> bool:
        # See OpenAIProvider._is_retryable: a rejected key is terminal.
        if isinstance(exc, (LLMRateLimitError, LLMAuthError)):
            return False
        return isinstance(exc, (LLMProviderError, LLMTimeoutError))

    @classmethod
    def is_multimodal_embedding_model(cls, model: str) -> bool:
        """Return whether *model* maps text and images into one space."""
        name = (model or "").strip().lower()
        if not name:
            return False
        return any(marker in name for marker in cls._MULTIMODAL_EMBEDDING_MARKERS)

    def _embed_content_config(self) -> Any:
        if types is None:
            _raise_missing_sdk()
        config_kwargs: dict[str, Any] = {"task_type": "SEMANTIC_SIMILARITY"}
        if self._embedding_output_dimensionality is not None:
            config_kwargs["output_dimensionality"] = self._embedding_output_dimensionality
        return types.EmbedContentConfig(**config_kwargs)

    @staticmethod
    def _embedding_values(response: Any) -> list[float]:
        embeddings = getattr(response, "embeddings", None) or []
        if not embeddings:
            return []
        values = getattr(embeddings[0], "values", None)
        if values is None:
            return []
        return list(values)

    async def embed(self, text: str, *, model: str = "gemini-embedding-001") -> list[float]:
        """Get text embedding using Gemini's embedding model.

        Args:
            text: Text to embed.
            model: Embedding model name (default: gemini-embedding-001).

        Returns:
            Embedding vector (dimension depends on model / config).
        """
        if types is None:
            _raise_missing_sdk()
        response = await self._client.aio.models.embed_content(
            model=model,
            contents=text,
            config=self._embed_content_config(),
        )
        return self._embedding_values(response)

    async def embed_image(
        self,
        image_bytes: bytes,
        *,
        mime_type: str = "image/jpeg",
        model: str = "gemini-embedding-2",
    ) -> list[float]:
        """Get image-only embedding (Gemini Embedding 2 multimodal space).

        Requires a multimodal embedding model. Returns ``[]`` when the
        model is text-only so callers can degrade without raising.
        """
        if types is None:
            _raise_missing_sdk()
        if not self.is_multimodal_embedding_model(model):
            return []
        if not image_bytes:
            return []
        part = types.Part.from_bytes(
            data=image_bytes,
            mime_type=(mime_type or "image/jpeg").strip() or "image/jpeg",
        )
        response = await self._client.aio.models.embed_content(
            model=model,
            contents=part,
            config=self._embed_content_config(),
        )
        return self._embedding_values(response)

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        chunks: list[str] = []
        for message in messages:
            content = message["content"].strip()
            if not content:
                continue
            role = message["role"].upper()
            chunks.append(f"[{role}]\n{content}")
        return "\n\n".join(chunks)
