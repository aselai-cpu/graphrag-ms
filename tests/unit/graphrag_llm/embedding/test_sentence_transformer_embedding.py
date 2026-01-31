# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for SentenceTransformerEmbedding."""

# ruff: noqa: SLF001

import sys
from unittest.mock import MagicMock, Mock, patch

import pytest
from graphrag_llm.config import ModelConfig

# Mock sentence_transformers module before importing our module
mock_sentence_transformers = MagicMock()
sys.modules["sentence_transformers"] = mock_sentence_transformers


class TestSentenceTransformerEmbedding:
    """Test suite for SentenceTransformerEmbedding initialization."""

    @pytest.fixture
    def mock_config(self):
        """Fixture providing a mock ModelConfig."""
        return ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

    @pytest.fixture
    def mock_dependencies(self):
        """Fixture providing mocked dependencies."""
        return {
            "tokenizer": Mock(),
            "metrics_store": Mock(),
            "cache_key_creator": Mock(),
        }

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    @patch("graphrag_llm.embedding.sentence_transformer_embedding._detect_device")
    def test_initialization_with_default_device(
        self,
        mock_detect_device,
        mock_create_base,
        mock_middleware,
        mock_config,
        mock_dependencies,
    ):
        """Test SentenceTransformerEmbedding initializes with default device detection.

        Arrange: Mock SentenceTransformer and device detection
        Act: Create SentenceTransformerEmbedding instance
        Assert: Device is detected and model is loaded
        """
        # Arrange
        mock_detect_device.return_value = "cuda"
        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            **mock_dependencies,
        )

        # Assert
        assert embedding is not None
        assert embedding._device == "cuda"
        assert embedding._model == mock_model
        mock_detect_device.assert_called_once()

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_initialization_with_explicit_cuda(
        self, mock_create_base, mock_middleware, mock_config, mock_dependencies
    ):
        """Test initialization with explicit CUDA device."""
        # Arrange
        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            device="cuda",
            **mock_dependencies,
        )

        # Assert
        assert embedding._device == "cuda"

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_initialization_with_explicit_cpu(
        self, mock_create_base, mock_middleware, mock_config, mock_dependencies
    ):
        """Test initialization with explicit CPU device."""
        # Arrange
        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            device="cpu",
            **mock_dependencies,
        )

        # Assert
        assert embedding._device == "cpu"

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_initialization_with_custom_batch_size(
        self, mock_create_base, mock_middleware, mock_config, mock_dependencies
    ):
        """Test initialization with custom batch size."""
        # Arrange
        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            device="cpu",
            batch_size=64,
            **mock_dependencies,
        )

        # Assert
        assert embedding._batch_size == 64

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_initialization_with_normalize_embeddings_false(
        self, mock_create_base, mock_middleware, mock_config, mock_dependencies
    ):
        """Test initialization with normalize_embeddings=False."""
        # Arrange
        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            device="cpu",
            normalize_embeddings=False,
            **mock_dependencies,
        )

        # Assert
        assert embedding._normalize_embeddings is False

    def test_initialization_raises_import_error_when_package_missing(
        self, mock_config, mock_dependencies
    ):
        """Test that ImportError is raised with helpful message when package missing."""
        # Remove the mocked module temporarily
        import importlib

        import graphrag_llm.embedding.sentence_transformer_embedding

        original_module = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = None  # type: ignore

        try:
            # Reload to trigger the import error
            importlib.reload(graphrag_llm.embedding.sentence_transformer_embedding)

            # Import the class (won't raise yet)
            from graphrag_llm.embedding.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            # This should raise ImportError during instantiation
            with pytest.raises(ImportError, match="sentence-transformers"):
                SentenceTransformerEmbedding(
                    model_id="test/model",
                    model_config=mock_config,
                    **mock_dependencies,
                )
        finally:
            # Restore the mocked module
            if original_module:
                sys.modules["sentence_transformers"] = original_module

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_stores_model_config_and_id(
        self, mock_create_base, mock_middleware, mock_config, mock_dependencies
    ):
        """Test that model_config and model_id are stored."""
        # Arrange
        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            device="cpu",
            **mock_dependencies,
        )

        # Assert
        assert embedding._model_id == "test/model"
        assert embedding._model_config == mock_config

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_stores_dependencies(
        self, mock_create_base, mock_middleware, mock_config, mock_dependencies
    ):
        """Test that all dependencies are stored."""
        # Arrange
        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            device="cpu",
            **mock_dependencies,
        )

        # Assert
        assert embedding._tokenizer == mock_dependencies["tokenizer"]
        assert embedding._metrics_store == mock_dependencies["metrics_store"]
        assert embedding._cache_key_creator == mock_dependencies["cache_key_creator"]


class TestDetectDevice:
    """Test suite for _detect_device helper function."""

    def test_detect_device_cuda_available(self):
        """Test that CUDA is detected when available."""
        # Arrange
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        sys.modules["torch"] = mock_torch

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _detect_device,
        )

        device = _detect_device()

        # Assert
        assert device == "cuda"

    def test_detect_device_mps_available(self):
        """Test that MPS is detected when CUDA unavailable but MPS available."""
        # Arrange
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        sys.modules["torch"] = mock_torch

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _detect_device,
        )

        device = _detect_device()

        # Assert
        assert device == "mps"

    def test_detect_device_cpu_fallback(self):
        """Test that CPU is used when neither CUDA nor MPS available."""
        # Arrange
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        sys.modules["torch"] = mock_torch

        # Act
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _detect_device,
        )

        device = _detect_device()

        # Assert
        assert device == "cpu"

    def test_detect_device_no_torch(self):
        """Test that CPU is used when torch is not available."""
        # Arrange
        import importlib

        import graphrag_llm.embedding.sentence_transformer_embedding

        original_torch = sys.modules.get("torch")
        sys.modules["torch"] = None  # type: ignore

        try:
            # Need to reload to test import failure
            importlib.reload(graphrag_llm.embedding.sentence_transformer_embedding)

            from graphrag_llm.embedding.sentence_transformer_embedding import (
                _detect_device,
            )

            # Act
            device = _detect_device()

            # Assert
            assert device == "cpu"
        finally:
            # Restore
            if original_torch:
                sys.modules["torch"] = original_torch
            # Reload again to restore
            importlib.reload(graphrag_llm.embedding.sentence_transformer_embedding)


class TestCreateBaseEmbeddings:
    """Test suite for _create_base_embeddings helper function."""

    def test_returns_two_callables(self):
        """Test that _create_base_embeddings returns two callable functions."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _create_base_embeddings,
        )

        mock_model = Mock()
        mock_embeddings = Mock()
        mock_embeddings.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = [mock_embeddings]
        mock_model.model_card_data.model_name = "test-model"

        # Act
        sync_fn, async_fn = _create_base_embeddings(
            model=mock_model,
            device="cpu",
            batch_size=32,
            normalize_embeddings=True,
        )

        # Assert
        assert callable(sync_fn)
        assert callable(async_fn)

    def test_sync_embedding_with_list_input(self):
        """Test synchronous embedding generation with list input."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _create_base_embeddings,
        )

        mock_model = Mock()
        mock_embedding1 = Mock()
        mock_embedding1.tolist.return_value = [0.1, 0.2, 0.3]
        mock_embedding2 = Mock()
        mock_embedding2.tolist.return_value = [0.4, 0.5, 0.6]
        mock_model.encode.return_value = [mock_embedding1, mock_embedding2]
        mock_model.model_card_data.model_name = "test-model"

        sync_fn, _ = _create_base_embeddings(
            model=mock_model,
            device="cpu",
            batch_size=32,
            normalize_embeddings=True,
        )

        # Act
        result = sync_fn(input=["text1", "text2"])

        # Assert
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.data[0].index == 0
        assert result.data[1].embedding == [0.4, 0.5, 0.6]
        assert result.data[1].index == 1
        assert result.model == "test-model"
        assert result.usage is not None

    def test_sync_embedding_with_string_input(self):
        """Test synchronous embedding generation with string input."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _create_base_embeddings,
        )

        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = [mock_embedding]
        mock_model.model_card_data.model_name = "test-model"

        sync_fn, _ = _create_base_embeddings(
            model=mock_model,
            device="cpu",
            batch_size=32,
            normalize_embeddings=True,
        )

        # Act
        result = sync_fn(input="single text")  # type: ignore

        # Assert
        assert len(result.data) == 1
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        mock_model.encode.assert_called_once()

    def test_sync_embedding_passes_parameters_to_model(self):
        """Test that sync function passes correct parameters to model.encode."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _create_base_embeddings,
        )

        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = [mock_embedding]
        mock_model.model_card_data.model_name = "test-model"

        sync_fn, _ = _create_base_embeddings(
            model=mock_model,
            device="cuda",
            batch_size=64,
            normalize_embeddings=False,
        )

        # Act
        sync_fn(input=["test"])

        # Assert
        mock_model.encode.assert_called_once_with(
            ["test"],
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=False,
            device="cuda",
        )

    @pytest.mark.asyncio
    async def test_async_embedding(self):
        """Test asynchronous embedding generation."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _create_base_embeddings,
        )

        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = [mock_embedding]
        mock_model.model_card_data.model_name = "test-model"

        _, async_fn = _create_base_embeddings(
            model=mock_model,
            device="cpu",
            batch_size=32,
            normalize_embeddings=True,
        )

        # Act
        result = await async_fn(input=["test"])

        # Assert
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        assert result.model == "test-model"

    def test_metrics_parameter_removed(self):
        """Test that metrics parameter is removed before processing."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            _create_base_embeddings,
        )

        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3]
        mock_model.encode.return_value = [mock_embedding]
        mock_model.model_card_data.model_name = "test-model"

        sync_fn, _ = _create_base_embeddings(
            model=mock_model,
            device="cpu",
            batch_size=32,
            normalize_embeddings=True,
        )

        # Act - should not raise even with metrics parameter
        result = sync_fn(input=["test"], metrics={})

        # Assert
        assert result is not None

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_metrics_store_property(self, mock_create_base, mock_middleware):
        """Test that metrics_store property returns the correct store."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )
        mock_metrics_store = Mock()

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=mock_metrics_store,
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = embedding.metrics_store

        # Assert
        assert result is mock_metrics_store

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_tokenizer_property(self, mock_create_base, mock_middleware):
        """Test that tokenizer property returns the correct tokenizer."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), Mock())

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )
        mock_tokenizer = Mock()

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=mock_tokenizer,
            metrics_store=Mock(),
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = embedding.tokenizer

        # Assert
        assert result is mock_tokenizer


class TestEmbeddingMethod:
    """Test suite for SentenceTransformerEmbedding.embedding() method."""

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_embedding_with_single_text(self, mock_create_base, mock_middleware):
        """Test embedding generation with single text input."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        # Mock the embedding function to return a response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_response.model = "test-model"
        mock_response.usage = Mock(prompt_tokens=1, total_tokens=1)

        mock_embedding_fn = Mock(return_value=mock_response)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (mock_embedding_fn, Mock())

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=Mock(),
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = embedding.embedding(input=["test text"])

        # Assert
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        mock_embedding_fn.assert_called_once()

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_embedding_with_multiple_texts(self, mock_create_base, mock_middleware):
        """Test embedding generation with multiple text inputs."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3], index=0),
            Mock(embedding=[0.4, 0.5, 0.6], index=1),
        ]

        mock_embedding_fn = Mock(return_value=mock_response)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (mock_embedding_fn, Mock())

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=Mock(),
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = embedding.embedding(input=["text1", "text2"])

        # Assert
        assert len(result.data) == 2
        mock_embedding_fn.assert_called_once_with(
            metrics=None, input=["text1", "text2"]
        )

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_embedding_with_metrics_tracking_enabled(
        self, mock_create_base, mock_middleware
    ):
        """Test that metrics are tracked when metrics_processor is provided."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]

        # Mock embedding function that populates metrics (as middleware would do)
        def mock_embedding_fn_with_metrics(**kwargs):
            metrics = kwargs.get("metrics")
            if metrics is not None:
                # Simulate middleware populating metrics
                metrics["total_tokens"] = 10
            return mock_response

        mock_embedding_fn = Mock(side_effect=mock_embedding_fn_with_metrics)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (mock_embedding_fn, Mock())

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )
        mock_metrics_store = Mock()
        mock_metrics_processor = Mock()

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=mock_metrics_store,
            metrics_processor=mock_metrics_processor,  # Metrics enabled
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = embedding.embedding(input=["test"])

        # Assert
        assert result is not None
        # Verify metrics were passed (not None)
        call_args = mock_embedding_fn.call_args
        assert call_args is not None
        assert "metrics" in call_args.kwargs
        assert call_args.kwargs["metrics"] is not None
        # Verify metrics_store.update_metrics was called with populated metrics
        mock_metrics_store.update_metrics.assert_called_once()
        # Verify the metrics dict was populated
        metrics_call_args = mock_metrics_store.update_metrics.call_args
        assert metrics_call_args.kwargs["metrics"]["total_tokens"] == 10

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_embedding_with_metrics_tracking_disabled(
        self, mock_create_base, mock_middleware
    ):
        """Test that metrics are not tracked when metrics_processor is None."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]

        mock_embedding_fn = Mock(return_value=mock_response)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (mock_embedding_fn, Mock())

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )
        mock_metrics_store = Mock()

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=mock_metrics_store,
            metrics_processor=None,  # Metrics disabled
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = embedding.embedding(input=["test"])

        # Assert
        assert result is not None
        # Verify metrics were None
        call_args = mock_embedding_fn.call_args
        assert call_args is not None
        assert "metrics" in call_args.kwargs
        assert call_args.kwargs["metrics"] is None
        # Verify metrics_store.update_metrics was NOT called
        mock_metrics_store.update_metrics.assert_not_called()

    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    def test_embedding_passes_kwargs_to_underlying_function(
        self, mock_create_base, mock_middleware
    ):
        """Test that additional kwargs are passed through to embedding function."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]

        mock_embedding_fn = Mock(return_value=mock_response)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (mock_embedding_fn, Mock())

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=Mock(),
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = embedding.embedding(input=["test"], custom_param="value")

        # Assert
        assert result is not None
        mock_embedding_fn.assert_called_once_with(
            metrics=None, input=["test"], custom_param="value"
        )


class TestEmbeddingAsyncMethod:
    """Test suite for SentenceTransformerEmbedding.embedding_async() method."""

    @pytest.mark.asyncio
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    async def test_embedding_async_with_single_text(
        self, mock_create_base, mock_middleware
    ):
        """Test async embedding generation with single text input."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        # Mock the embedding function to return a response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_response.model = "test-model"
        mock_response.usage = Mock(prompt_tokens=1, total_tokens=1)

        async def mock_async_embedding_fn(**kwargs):  # type: ignore  # noqa: RUF029
            return mock_response

        mock_embedding_async_fn = Mock(side_effect=mock_async_embedding_fn)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), mock_embedding_async_fn)

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=Mock(),
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = await embedding.embedding_async(input=["test text"])

        # Assert
        assert result.data[0].embedding == [0.1, 0.2, 0.3]
        mock_embedding_async_fn.assert_called_once()

    @pytest.mark.asyncio
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    async def test_embedding_async_with_multiple_texts(
        self, mock_create_base, mock_middleware
    ):
        """Test async embedding generation with multiple text inputs."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3], index=0),
            Mock(embedding=[0.4, 0.5, 0.6], index=1),
        ]

        async def mock_async_embedding_fn(**kwargs):  # type: ignore  # noqa: RUF029
            return mock_response

        mock_embedding_async_fn = Mock(side_effect=mock_async_embedding_fn)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), mock_embedding_async_fn)

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=Mock(),
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = await embedding.embedding_async(input=["text1", "text2"])

        # Assert
        assert len(result.data) == 2
        mock_embedding_async_fn.assert_called_once_with(
            metrics=None, input=["text1", "text2"]
        )

    @pytest.mark.asyncio
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    async def test_embedding_async_with_metrics_tracking_enabled(
        self, mock_create_base, mock_middleware
    ):
        """Test that metrics are tracked when metrics_processor is provided."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]

        # Mock async embedding function that populates metrics
        async def mock_async_embedding_fn_with_metrics(**kwargs):  # noqa: RUF029
            metrics = kwargs.get("metrics")
            if metrics is not None:
                # Simulate middleware populating metrics
                metrics["total_tokens"] = 10
            return mock_response

        mock_embedding_async_fn = Mock(side_effect=mock_async_embedding_fn_with_metrics)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), mock_embedding_async_fn)

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )
        mock_metrics_store = Mock()
        mock_metrics_processor = Mock()

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=mock_metrics_store,
            metrics_processor=mock_metrics_processor,  # Metrics enabled
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = await embedding.embedding_async(input=["test"])

        # Assert
        assert result is not None
        # Verify metrics were passed (not None)
        call_args = mock_embedding_async_fn.call_args
        assert call_args is not None
        assert "metrics" in call_args.kwargs
        assert call_args.kwargs["metrics"] is not None
        # Verify metrics_store.update_metrics was called with populated metrics
        mock_metrics_store.update_metrics.assert_called_once()
        # Verify the metrics dict was populated
        metrics_call_args = mock_metrics_store.update_metrics.call_args
        assert metrics_call_args.kwargs["metrics"]["total_tokens"] == 10

    @pytest.mark.asyncio
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    async def test_embedding_async_with_metrics_tracking_disabled(
        self, mock_create_base, mock_middleware
    ):
        """Test that metrics are not tracked when metrics_processor is None."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]

        async def mock_async_embedding_fn(**kwargs):  # type: ignore  # noqa: RUF029
            return mock_response

        mock_embedding_async_fn = Mock(side_effect=mock_async_embedding_fn)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), mock_embedding_async_fn)

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )
        mock_metrics_store = Mock()

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=mock_metrics_store,
            metrics_processor=None,  # Metrics disabled
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = await embedding.embedding_async(input=["test"])

        # Assert
        assert result is not None
        # Verify metrics were None
        call_args = mock_embedding_async_fn.call_args
        assert call_args is not None
        assert "metrics" in call_args.kwargs
        assert call_args.kwargs["metrics"] is None
        # Verify metrics_store.update_metrics was NOT called
        mock_metrics_store.update_metrics.assert_not_called()

    @pytest.mark.asyncio
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
    )
    @patch(
        "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
    )
    async def test_embedding_async_passes_kwargs_to_underlying_function(
        self, mock_create_base, mock_middleware
    ):
        """Test that additional kwargs are passed through to async embedding function."""
        # Arrange
        from graphrag_llm.embedding.sentence_transformer_embedding import (
            SentenceTransformerEmbedding,
        )

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]

        async def mock_async_embedding_fn(**kwargs):  # type: ignore  # noqa: RUF029
            return mock_response

        mock_embedding_async_fn = Mock(side_effect=mock_async_embedding_fn)
        mock_create_base.return_value = (Mock(), Mock())
        mock_middleware.return_value = (Mock(), mock_embedding_async_fn)

        mock_config = ModelConfig(
            model="all-MiniLM-L6-v2",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        embedding = SentenceTransformerEmbedding(
            model_id="test/model",
            model_config=mock_config,
            tokenizer=Mock(),
            metrics_store=Mock(),
            cache_key_creator=Mock(),
            device="cpu",
        )

        # Act
        result = await embedding.embedding_async(input=["test"], custom_param="value")

        # Assert
        assert result is not None
        mock_embedding_async_fn.assert_called_once_with(
            metrics=None, input=["test"], custom_param="value"
        )
