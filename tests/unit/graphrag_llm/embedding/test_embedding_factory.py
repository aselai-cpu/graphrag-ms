# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit tests for embedding factory."""

import sys
from unittest.mock import MagicMock, Mock, patch

from graphrag_llm.config import ModelConfig

# Mock sentence_transformers module before importing
mock_sentence_transformers = MagicMock()
sys.modules["sentence_transformers"] = mock_sentence_transformers


class TestEmbeddingFactory:
    """Test suite for embedding factory."""

    def test_create_embedding_with_sentence_transformer_type(self):
        """Test that factory creates SentenceTransformerEmbedding for sentence_transformer type."""
        # Arrange
        from graphrag_llm.embedding import create_embedding

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        config = ModelConfig(
            model="BAAI/bge-large-en-v1.5",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        # Act
        with (
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
            ) as mock_create_base,
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
            ) as mock_middleware,
        ):
            mock_create_base.return_value = (Mock(), Mock())
            mock_middleware.return_value = (Mock(), Mock())

            embedding = create_embedding(config)

            # Assert
            assert embedding is not None
            # Verify it's a SentenceTransformerEmbedding instance
            from graphrag_llm.embedding.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            assert isinstance(embedding, SentenceTransformerEmbedding)

    def test_create_embedding_with_sentence_transformer_initializes_successfully(self):
        """Test factory successfully initializes SentenceTransformerEmbedding with model_extra."""
        # Arrange
        from graphrag_llm.embedding import create_embedding

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        config = ModelConfig(
            model="BAAI/bge-large-en-v1.5",
            model_provider="sentence_transformer",
            type="sentence_transformer",
            model_extra={"device": "cuda", "batch_size": 64},  # type: ignore
        )

        # Act
        with (
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
            ) as mock_create_base,
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
            ) as mock_middleware,
        ):
            mock_create_base.return_value = (Mock(), Mock())
            mock_middleware.return_value = (Mock(), Mock())

            embedding = create_embedding(config)

            # Assert
            assert embedding is not None
            from graphrag_llm.embedding.sentence_transformer_embedding import (
                SentenceTransformerEmbedding,
            )

            assert isinstance(embedding, SentenceTransformerEmbedding)

    def test_create_embedding_with_litellm_type_still_works(self):
        """Test that existing LiteLLM type still works (backward compatibility)."""
        # Arrange
        from graphrag_llm.embedding import create_embedding

        config = ModelConfig(
            model="text-embedding-3-small",
            model_provider="openai",
            type="litellm",
            api_key="test-key",
        )

        # Act
        with patch("litellm.embedding") as mock_litellm_embedding:
            mock_response = Mock()
            mock_response.model_dump.return_value = {
                "data": [{"embedding": [0.1, 0.2], "index": 0, "object": "embedding"}],
                "model": "text-embedding-3-small",
                "object": "list",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }
            mock_litellm_embedding.return_value = mock_response

            embedding = create_embedding(config)

            # Assert
            assert embedding is not None
            from graphrag_llm.embedding.lite_llm_embedding import LiteLLMEmbedding

            assert isinstance(embedding, LiteLLMEmbedding)

    def test_create_embedding_registers_sentence_transformer_once(self):
        """Test that SentenceTransformer type is registered only once."""
        # Arrange
        from graphrag_llm.embedding import create_embedding
        from graphrag_llm.embedding.embedding_factory import embedding_factory

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        config = ModelConfig(
            model="BAAI/bge-large-en-v1.5",
            model_provider="sentence_transformer",
            type="sentence_transformer",
        )

        # Act
        with (
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
            ) as mock_create_base,
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
            ) as mock_middleware,
        ):
            mock_create_base.return_value = (Mock(), Mock())
            mock_middleware.return_value = (Mock(), Mock())

            # Create first instance
            embedding1 = create_embedding(config)
            assert embedding1 is not None

            # Check that sentence_transformer is now registered
            assert "sentence_transformer" in embedding_factory

            # Create second instance (should use existing registration)
            embedding2 = create_embedding(config)
            assert embedding2 is not None

    def test_create_embedding_with_metrics(self):
        """Test factory configures metrics for SentenceTransformerEmbedding."""
        # Arrange
        from graphrag_llm.config import MetricsConfig
        from graphrag_llm.embedding import create_embedding

        mock_model = Mock()
        mock_sentence_transformers.SentenceTransformer.return_value = mock_model

        config = ModelConfig(
            model="BAAI/bge-large-en-v1.5",
            model_provider="sentence_transformer",
            type="sentence_transformer",
            metrics=MetricsConfig(),
        )

        # Act
        with (
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding._create_base_embeddings"
            ) as mock_create_base,
            patch(
                "graphrag_llm.embedding.sentence_transformer_embedding.with_middleware_pipeline"
            ) as mock_middleware,
        ):
            mock_create_base.return_value = (Mock(), Mock())
            mock_middleware.return_value = (Mock(), Mock())

            embedding = create_embedding(config)

            # Assert
            assert embedding is not None
            # Verify metrics were configured
            assert embedding._metrics_store is not None  # noqa: SLF001  # type: ignore
            assert embedding._metrics_processor is not None  # noqa: SLF001  # type: ignore
