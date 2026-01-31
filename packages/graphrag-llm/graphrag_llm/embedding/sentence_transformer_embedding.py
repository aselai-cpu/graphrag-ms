# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""SentenceTransformer-based embedding generation."""

from typing import TYPE_CHECKING, Any, Unpack

from graphrag_llm.embedding.embedding import LLMEmbedding
from graphrag_llm.middleware import with_middleware_pipeline
from graphrag_llm.types import LLMEmbeddingResponse

if TYPE_CHECKING:
    from graphrag_cache import Cache, CacheKeyCreator

    from graphrag_llm.config import ModelConfig
    from graphrag_llm.metrics import MetricsProcessor, MetricsStore
    from graphrag_llm.rate_limit import RateLimiter
    from graphrag_llm.retry import Retry
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_llm.types import (
        AsyncLLMEmbeddingFunction,
        LLMEmbeddingArgs,
        LLMEmbeddingFunction,
        Metrics,
    )


class SentenceTransformerEmbedding(LLMEmbedding):
    """Local embedding generation using SentenceTransformers.

    This class provides a local, free alternative to API-based embeddings
    using the sentence-transformers library. It supports CUDA, CPU, and MPS devices.

    Attributes
    ----------
    _model : SentenceTransformer
        The loaded SentenceTransformer model instance.
    _device : str
        The device being used: "cuda", "cpu", or "mps".
    _batch_size : int
        The batch size for encoding operations.
    _normalize_embeddings : bool
        Whether to L2-normalize the generated embeddings.

    Examples
    --------
    >>> from graphrag_llm.embedding import (
    ...     create_embedding,
    ... )
    >>> from graphrag_llm.config import (
    ...     ModelConfig,
    ... )
    >>> config = ModelConfig(
    ...     model="all-MiniLM-L6-v2",
    ...     model_provider="sentence_transformer",
    ...     type="sentence_transformer",
    ... )
    >>> embedding = create_embedding(
    ...     config
    ... )
    >>> result = embedding.embedding(
    ...     input=[
    ...         "Hello world"
    ...     ]
    ... )
    >>> len(
    ...     result.data[0][
    ...         "embedding"
    ...     ]
    ... )
    384
    """

    _model_config: "ModelConfig"
    _model_id: str
    _model: Any  # SentenceTransformer model
    _device: str
    _batch_size: int
    _normalize_embeddings: bool
    _track_metrics: bool
    _metrics_store: "MetricsStore"
    _metrics_processor: "MetricsProcessor | None"
    _cache: "Cache | None"
    _cache_key_creator: "CacheKeyCreator"
    _tokenizer: "Tokenizer"
    _rate_limiter: "RateLimiter | None"
    _retrier: "Retry | None"
    _embedding: "LLMEmbeddingFunction"
    _embedding_async: "AsyncLLMEmbeddingFunction"

    def __init__(
        self,
        *,
        model_id: str,
        model_config: "ModelConfig",
        tokenizer: "Tokenizer",
        metrics_store: "MetricsStore",
        metrics_processor: "MetricsProcessor | None" = None,
        rate_limiter: "RateLimiter | None" = None,
        retrier: "Retry | None" = None,
        cache: "Cache | None" = None,
        cache_key_creator: "CacheKeyCreator",
        device: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize SentenceTransformerEmbedding.

        Parameters
        ----------
        model_id : str
            The model identifier.
        model_config : ModelConfig
            The model configuration.
        tokenizer : Tokenizer
            The tokenizer instance.
        metrics_store : MetricsStore
            The metrics store instance.
        metrics_processor : MetricsProcessor | None, default=None
            Optional metrics processor.
        rate_limiter : RateLimiter | None, default=None
            Optional rate limiter (not typically needed for local).
        retrier : Retry | None, default=None
            Optional retry strategy.
        cache : Cache | None, default=None
            Optional cache instance.
        cache_key_creator : CacheKeyCreator
            Cache key creator function.
        device : str | None, default=None
            Device to run model on: "cuda", "cpu", or "mps".
            If None, automatically detects best available device.
        batch_size : int, default=32
            Batch size for encoding operations.
        normalize_embeddings : bool, default=True
            Whether to L2-normalize embeddings.
        **kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        ImportError
            If sentence-transformers package is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            msg = (
                "sentence-transformers package not installed. "
                "Install it with: pip install sentence-transformers\n"
                "Or install graphrag with: pip install graphrag[local-embeddings]"
            )
            raise ImportError(msg) from e

        self._model_id = model_id
        self._model_config = model_config
        self._tokenizer = tokenizer
        self._metrics_store = metrics_store
        self._metrics_processor = metrics_processor
        self._track_metrics = metrics_processor is not None
        self._cache = cache
        self._cache_key_creator = cache_key_creator
        self._rate_limiter = rate_limiter
        self._retrier = retrier
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

        # Detect device if not specified
        self._device = device or _detect_device()

        # Load the SentenceTransformer model
        model_name = model_config.model
        self._model = SentenceTransformer(model_name, device=self._device)

        # Create base embedding functions
        self._embedding, self._embedding_async = _create_base_embeddings(
            model=self._model,
            device=self._device,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
        )

        # Wrap with middleware pipeline
        self._embedding, self._embedding_async = with_middleware_pipeline(
            model_config=self._model_config,
            model_fn=self._embedding,
            async_model_fn=self._embedding_async,
            request_type="embedding",
            cache=self._cache,
            cache_key_creator=self._cache_key_creator,
            tokenizer=self._tokenizer,
            metrics_processor=self._metrics_processor,
            rate_limiter=self._rate_limiter,
            retrier=self._retrier,
        )

    def embedding(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Generate embeddings synchronously.

        Parameters
        ----------
        **kwargs : Unpack[LLMEmbeddingArgs]
            Keyword arguments including input texts.

        Returns
        -------
        LLMEmbeddingResponse
            Generated embeddings with usage statistics.
        """
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return self._embedding(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    async def embedding_async(
        self, /, **kwargs: Unpack["LLMEmbeddingArgs"]
    ) -> "LLMEmbeddingResponse":
        """Generate embeddings asynchronously.

        Parameters
        ----------
        **kwargs : Unpack[LLMEmbeddingArgs]
            Keyword arguments including input texts.

        Returns
        -------
        LLMEmbeddingResponse
            Generated embeddings with usage statistics.
        """
        request_metrics: Metrics | None = kwargs.pop("metrics", None) or {}
        if not self._track_metrics:
            request_metrics = None

        try:
            return await self._embedding_async(metrics=request_metrics, **kwargs)
        finally:
            if request_metrics:
                self._metrics_store.update_metrics(metrics=request_metrics)

    @property
    def metrics_store(self) -> "MetricsStore":
        """Get metrics store.

        Returns
        -------
        MetricsStore
            The metrics store instance.
        """
        return self._metrics_store

    @property
    def tokenizer(self) -> "Tokenizer":
        """Get tokenizer.

        Returns
        -------
        Tokenizer
            The tokenizer instance.
        """
        return self._tokenizer


def _detect_device() -> str:
    """Detect the best available device for embedding generation.

    Returns
    -------
    str
        One of "cuda", "mps", or "cpu".
    """
    try:
        import torch  # type: ignore
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _create_base_embeddings(
    *,
    model: Any,
    device: str,
    batch_size: int,
    normalize_embeddings: bool,
) -> tuple["LLMEmbeddingFunction", "AsyncLLMEmbeddingFunction"]:
    """Create base embedding functions for SentenceTransformer.

    Parameters
    ----------
    model : SentenceTransformer
        The loaded model instance.
    device : str
        The device to use.
    batch_size : int
        Batch size for encoding.
    normalize_embeddings : bool
        Whether to normalize embeddings.

    Returns
    -------
    tuple[LLMEmbeddingFunction, AsyncLLMEmbeddingFunction]
        Sync and async embedding functions.
    """
    import asyncio

    from openai.types.create_embedding_response import Usage  # type: ignore
    from openai.types.embedding import Embedding  # type: ignore

    def _base_embedding(**kwargs: Any) -> LLMEmbeddingResponse:
        kwargs.pop("metrics", None)  # Remove metrics if present
        input_texts: list[str] = kwargs.get("input", [])

        if isinstance(input_texts, str):
            input_texts = [input_texts]

        # Generate embeddings
        embeddings = model.encode(
            input_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize_embeddings,
            device=device,
        )

        # Format response to match OpenAI structure
        data = [
            Embedding(embedding=emb.tolist(), index=i, object="embedding")
            for i, emb in enumerate(embeddings)
        ]

        token_count = sum(len(t.split()) for t in input_texts)

        return LLMEmbeddingResponse(
            data=data,
            model=model.model_card_data.model_name or "sentence-transformer",
            object="list",
            usage=Usage(
                prompt_tokens=token_count,
                total_tokens=token_count,
            ),
        )

    async def _base_embedding_async(**kwargs: Any) -> LLMEmbeddingResponse:
        # Run synchronous embedding in thread pool
        return await asyncio.to_thread(_base_embedding, **kwargs)  # type: ignore

    return _base_embedding, _base_embedding_async
