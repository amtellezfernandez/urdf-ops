"""Compute factory for creating compute backends from configuration.

This module provides a simple factory function to instantiate compute backends
based on configuration, making it easy to switch between local and cloud.

Usage:
    from backend.robotops.compute_factory import get_compute, ComputeConfig

    config = ComputeConfig(type="modal", api_key="...")
    compute = get_compute(config)
    job_id = await compute.launch("train.py", {...})
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from backend.robotops.compute_protocol import ComputeBackend
from backend.robotops.compute.local_compute import LocalCompute
from backend.robotops.compute.macrodata_compute import MacrodataCompute
from backend.robotops.compute.modal_compute import ModalCompute
from backend.robotops.compute.runpod_compute import RunPodCompute

logger = logging.getLogger(__name__)


ComputeType = Literal["local", "modal", "runpod", "macrodata"]

SECRET_CACHE_FIELD_MARKERS = (
    "api_key",
    "token",
    "secret",
    "password",
    "credential",
    "private_key",
    "access_key",
)


class ComputeConfig(BaseModel):
    """Configuration for compute backend.

    Attributes:
        type: Compute type - "local", "modal", "runpod", or "macrodata"

        Local-specific:
            python_path: Path to Python interpreter

        Modal-specific:
            api_key: Modal API key
            default_gpu: Default GPU type

        RunPod-specific:
            api_key: RunPod API key
            default_gpu: Default GPU type
            use_spot: Whether to use spot instances

        Common:
            output_dir: Directory for outputs
    """

    type: ComputeType = Field(default="local", description="Compute type")

    # API keys (for cloud providers)
    api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud provider",
    )

    # GPU config
    default_gpu: Optional[str] = Field(
        default=None,
        description="Default GPU type (e.g., 'T4', 'A100-40GB')",
    )

    # Local-specific
    python_path: Optional[str] = Field(
        default=None,
        description="Path to Python interpreter (local only)",
    )

    # RunPod-specific
    use_spot: bool = Field(
        default=True,
        description="Use spot instances for cost savings (RunPod)",
    )

    # Common
    output_dir: str = Field(
        default="./outputs",
        description="Directory for job outputs",
    )

    model_config = ConfigDict(extra="allow")


# Registry of available compute backends
COMPUTE_REGISTRY: Dict[str, type] = {
    "local": LocalCompute,
    "modal": ModalCompute,
    "runpod": RunPodCompute,
    "macrodata": MacrodataCompute,
}

# Cache of compute backend instances (to preserve job state)
_COMPUTE_INSTANCES: Dict[str, ComputeBackend] = {}


def _hash_secret(value: str | None) -> str | None:
    if not value:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_secret_cache_field(field_name: str) -> bool:
    normalized = field_name.lower()
    return any(marker in normalized for marker in SECRET_CACHE_FIELD_MARKERS)


def _sanitize_compute_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        return _sanitize_compute_cache_payload(value)
    if isinstance(value, list):
        return [_sanitize_compute_cache_value(item) for item in value]
    return value


def _sanitize_compute_cache_payload(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in payload.items():
        if _is_secret_cache_field(key):
            if isinstance(value, (dict, list)):
                sanitized[key] = _sanitize_compute_cache_value(value)
            elif value:
                sanitized[f"{key}_sha256"] = _hash_secret(str(value))
            continue
        sanitized[key] = _sanitize_compute_cache_value(value)
    return sanitized


def _build_compute_cache_key(config: ComputeConfig, compute_type: str) -> str:
    cache_payload = _sanitize_compute_cache_payload(config.model_dump(mode="json"))
    return json.dumps(
        {"type": compute_type, "config": cache_payload},
        sort_keys=True,
        separators=(",", ":"),
    )


def get_compute(
    config: Union[ComputeConfig, Dict[str, Any], None] = None,
) -> ComputeBackend:
    """Get or create a compute backend from configuration.

    Compute backends are cached to preserve job state between calls.

    Args:
        config: Compute configuration. Can be:
            - ComputeConfig instance
            - Dictionary with compute settings
            - None (returns LocalCompute)

    Returns:
        A ComputeBackend instance (cached)

    Examples:
        # Using ComputeConfig
        compute = get_compute(ComputeConfig(type="modal", api_key="..."))

        # Using dict
        compute = get_compute({"type": "runpod", "api_key": "...", "use_spot": True})

        # Default (local)
        compute = get_compute()
    """
    if config is None:
        config = ComputeConfig(type="local")
    elif isinstance(config, dict):
        config = ComputeConfig(**config)

    compute_type = config.type

    compute_cls = COMPUTE_REGISTRY.get(compute_type)

    if compute_cls is None:
        logger.warning(f"Unknown compute type: {compute_type}, using local")
        compute_type = "local"
        compute_cls = LocalCompute

    cache_key = _build_compute_cache_key(config, compute_type)

    # Return cached instance if available.
    if cache_key in _COMPUTE_INSTANCES:
        return _COMPUTE_INSTANCES[cache_key]

    # Build kwargs based on compute type
    kwargs: Dict[str, Any] = {"output_dir": config.output_dir}

    if compute_type == "local":
        if config.python_path:
            kwargs["python_path"] = config.python_path

    elif compute_type == "modal":
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.default_gpu:
            kwargs["default_gpu"] = config.default_gpu

    elif compute_type == "runpod":
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.default_gpu:
            kwargs["default_gpu"] = config.default_gpu
        kwargs["use_spot"] = config.use_spot

    elif compute_type == "macrodata":
        if config.api_key:
            kwargs["api_key"] = config.api_key

    # Include any extra config fields
    extra_fields = set(config.model_dump().keys()) - set(ComputeConfig.model_fields.keys())
    for field in extra_fields:
        kwargs[field] = getattr(config, field)

    logger.info(f"Creating {compute_type} compute backend (cached)")
    instance = compute_cls(**kwargs)
    _COMPUTE_INSTANCES[cache_key] = instance
    return instance


def register_compute(name: str, compute_cls: type) -> None:
    """Register a custom compute backend implementation.

    Args:
        name: Compute type name (used in config)
        compute_cls: Compute class implementing ComputeBackend protocol

    Example:
        class LambdaLabsCompute:
            name = "lambda"
            # ... implement ComputeBackend protocol

        register_compute("lambda", LambdaLabsCompute)
    """
    if not isinstance(compute_cls, type):
        raise TypeError(f"compute_cls must be a class, got {type(compute_cls)}")

    COMPUTE_REGISTRY[name] = compute_cls
    logger.info(f"Registered compute backend: {name}")


def list_available_compute() -> list[str]:
    """Return list of available compute types."""
    return list(COMPUTE_REGISTRY.keys())


async def get_all_available_instances() -> Dict[str, list]:
    """Get available GPU instances from all backends.

    Returns:
        Dict mapping backend name to list of available instances
    """
    results = {}

    # Local
    local = LocalCompute()
    results["local"] = await local.get_available_instances()

    # Cloud providers (if available)
    for name in ["modal", "runpod", "macrodata"]:
        try:
            compute = get_compute(ComputeConfig(type=name))
            results[name] = await compute.get_available_instances()
        except Exception as e:
            logger.debug(f"Could not get instances from {name}: {e}")
            results[name] = []

    return results
