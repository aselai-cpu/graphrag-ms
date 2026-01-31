# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Configuration for dark mode execution."""

from pydantic import BaseModel, Field

from graphrag.config.defaults import graphrag_config_defaults


class DarkModeComparisonConfig(BaseModel):
    """Configuration for dark mode comparison thresholds."""

    entity_match_threshold: float = Field(
        description="Minimum entity F1 score to pass comparison (0.0-1.0).",
        default=graphrag_config_defaults.dark_mode.entity_match_threshold,
        ge=0.0,
        le=1.0,
    )
    community_match_threshold: float = Field(
        description="Minimum community assignment match rate (0.0-1.0).",
        default=graphrag_config_defaults.dark_mode.community_match_threshold,
        ge=0.0,
        le=1.0,
    )
    degree_tolerance: float = Field(
        description="Maximum relative difference in node degrees (0.0-1.0).",
        default=graphrag_config_defaults.dark_mode.degree_tolerance,
        ge=0.0,
        le=1.0,
    )


class DarkModeCutoverConfig(BaseModel):
    """Configuration for dark mode cutover criteria."""

    min_operations: int = Field(
        description="Minimum number of operations to validate before cutover.",
        default=graphrag_config_defaults.dark_mode.min_operations,
        ge=1,
    )
    max_error_rate: float = Field(
        description="Maximum shadow error rate allowed for cutover (0.0-1.0).",
        default=graphrag_config_defaults.dark_mode.max_error_rate,
        ge=0.0,
        le=1.0,
    )
    min_pass_rate: float = Field(
        description="Minimum comparison pass rate required for cutover (0.0-1.0).",
        default=graphrag_config_defaults.dark_mode.min_pass_rate,
        ge=0.0,
        le=1.0,
    )
    max_latency_ratio: float = Field(
        description="Maximum shadow/primary latency ratio allowed for cutover.",
        default=graphrag_config_defaults.dark_mode.max_latency_ratio,
        ge=1.0,
    )


class DarkModeConfig(BaseModel):
    """Configuration section for dark mode execution.

    Dark mode runs primary and shadow graph backends in parallel,
    comparing results to validate shadow backend before production cutover.
    """

    enabled: bool = Field(
        description="Whether dark mode execution is enabled.",
        default=graphrag_config_defaults.dark_mode.enabled,
    )
    primary_backend: str = Field(
        description="Primary backend name (serves production traffic).",
        default=graphrag_config_defaults.dark_mode.primary_backend,
    )
    shadow_backend: str = Field(
        description="Shadow backend name (validation only, does not affect results).",
        default=graphrag_config_defaults.dark_mode.shadow_backend,
    )
    log_path: str | None = Field(
        description="Path to write metrics log file (JSON lines format). If None, logs to memory only.",
        default=graphrag_config_defaults.dark_mode.log_path,
    )
    comparison: DarkModeComparisonConfig = Field(
        description="Comparison thresholds for validating shadow results.",
        default_factory=DarkModeComparisonConfig,
    )
    cutover_criteria: DarkModeCutoverConfig = Field(
        description="Criteria for determining cutover readiness.",
        default_factory=DarkModeCutoverConfig,
    )
