from __future__ import annotations as _annotations

import os

import httpx

from pydantic_ai.models import cached_async_http_client
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.profiles.nvidia_nim import nvidia_model_profile
from pydantic_ai.providers import Provider

try:
    from openai import AsyncOpenAI
except ImportError as _import_error:  # pragma: no cover
    raise ImportError(
        'Please install the `openai` package to use the Nvidia NIM provider, '
        'you can use the `openai` optional group — `pip install "pydantic-ai-slim[openai]"`'
    ) from _import_error


class NvidiaNIMProvider(Provider[AsyncOpenAI]):
    """Provider for Nvidia NIM, which exposes an OpenAI-compatible API.

    This provider configures the `AsyncOpenAI` client to connect to a specified
    Nvidia NIM endpoint.
    """

    @property
    def name(self) -> str:
        """The name of the provider."""
        return 'nvidia'

    @property
    def base_url(self) -> str:
        """The base URL of the Nvidia NIM API endpoint."""
        return str(self.client.base_url)

    @property
    def client(self) -> AsyncOpenAI:
        """The underlying `AsyncOpenAI` client used for making API requests."""
        return self._client

    def model_profile(self, model_name: str) -> ModelProfile | None:
        """Retrieves the model profile for a given Nvidia NIM model name.

        Args:
            model_name: The name of the model.

        Returns:
            The corresponding model profile.
        """
        return nvidia_model_profile(model_name)

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        openai_client: AsyncOpenAI | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Create a new Nvidia NIM provider.

        Args:
            base_url: The base URL for the Nvidia NIM endpoint. If not provided, the
                `NVIDIA_BASE_URL` environment variable will be used. Defaults to
                Nvidia's public API endpoint if neither is set.
            api_key: The API key for authentication. If not provided, the
                `NVIDIA_API_KEY` environment variable will be used.
            openai_client: An existing `AsyncOpenAI` client to use. If provided,
                `base_url`, `api_key`, and `http_client` must be `None`.
            http_client: An existing `httpx.AsyncClient` to use for HTTP requests.
        """
        # Set default base_url if not provided via argument or environment variable
        if base_url is None and 'NVIDIA_BASE_URL' not in os.environ:
            base_url = 'https://integrate.api.nvidia.com/v1'

        # The OpenAI client requires a non-empty API key. If one isn't provided
        # (e.g., for a local NIM without auth), use a placeholder.
        if api_key is None and 'NVIDIA_API_KEY' not in os.environ and openai_client is None:
            api_key = 'api-key-not-required'

        # Resolve environment variables if arguments are not provided
        resolved_base_url = base_url or os.environ.get('NVIDIA_BASE_URL')
        resolved_api_key = api_key or os.environ.get('NVIDIA_API_KEY')

        if openai_client is not None:
            if any([base_url, http_client, api_key]):
                raise ValueError('Cannot provide `openai_client` along with `base_url`, `http_client`, or `api_key`.')
            self._client = openai_client
        else:
            # Use a shared, cached HTTP client if a specific one isn't provided
            if http_client is None:
                http_client = cached_async_http_client(provider='nvidia')

            self._client = AsyncOpenAI(
                base_url=resolved_base_url, api_key=resolved_api_key, http_client=http_client, timeout=30.0
            )
