from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Mapping

from backend.robotops.compute.cloud_control_params import (
    TRAINING_CLOUD_CONTROL_EXECUTOR_TYPE,
    TRAINING_CLOUD_CONTROL_INLINE_PAYLOAD_MAX_BYTES,
    TRAINING_CLOUD_CONTROL_PAYLOAD_FORMAT,
    TRAINING_CLOUD_CONTROL_REDACTION_PLACEHOLDER,
    TRAINING_CLOUD_CONTROL_SECRET_MIN_REDACTION_CHARS,
)


@dataclass(frozen=True, slots=True)
class TrainingCloudRuntimeConfig:
    num_workers: int
    heartbeat_interval_seconds: int
    cpus_per_worker: int | None = None
    mem_mb_per_worker: int | None = None
    gpus_per_worker: int | None = None
    gpu_type: str | None = None

    def __post_init__(self) -> None:
        _require_positive("num_workers", self.num_workers)
        _require_positive("heartbeat_interval_seconds", self.heartbeat_interval_seconds)
        _require_optional_positive("cpus_per_worker", self.cpus_per_worker)
        _require_optional_positive("mem_mb_per_worker", self.mem_mb_per_worker)
        _require_optional_positive("gpus_per_worker", self.gpus_per_worker)
        if self.gpus_per_worker is not None and self.gpu_type is None:
            raise ValueError("gpu_type is required when gpus_per_worker is set")
        if self.gpu_type is not None and not self.gpu_type.strip():
            raise ValueError("gpu_type must be non-empty")
        if self.gpu_type is not None and self.gpus_per_worker is None:
            raise ValueError("gpus_per_worker is required when gpu_type is set")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_workers": self.num_workers,
            "heartbeat_interval_seconds": self.heartbeat_interval_seconds,
        }
        if self.cpus_per_worker is not None:
            payload["cpus_per_worker"] = self.cpus_per_worker
        if self.mem_mb_per_worker is not None:
            payload["mem_mb_per_worker"] = self.mem_mb_per_worker
        if self.gpus_per_worker is not None:
            payload["gpus_per_worker"] = self.gpus_per_worker
        if self.gpu_type is not None:
            payload["gpu_type"] = self.gpu_type
        return payload


@dataclass(frozen=True, slots=True)
class TrainingCloudPayload:
    format: str
    body: dict[str, Any]
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "body": self.body,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class TrainingCloudRunCreateRequest:
    provider: str
    name: str
    runtime: TrainingCloudRuntimeConfig
    payload: TrainingCloudPayload
    manifest: dict[str, Any] | None = None
    secrets: dict[str, str] | None = None
    env: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        request: dict[str, Any] = {
            "name": self.name,
            "executor": {
                "type": TRAINING_CLOUD_CONTROL_EXECUTOR_TYPE,
                "provider": self.provider,
            },
            "runtime": self.runtime.to_dict(),
            "payload": self.payload.to_dict(),
        }
        if self.manifest is not None:
            request["manifest"] = self.manifest
        if self.secrets:
            request["secrets"] = self.secrets
        if self.env:
            request["env"] = self.env
        return request


def resolve_training_cloud_env_values(
    values: Mapping[str, object | None] | None,
) -> dict[str, str] | None:
    if not values:
        return None
    resolved: dict[str, str] = {}
    for name, value in values.items():
        if value is None:
            env_value = os.environ.get(name)
            if env_value is None:
                raise ValueError(
                    f"cloud env {name!r} is not present in the environment"
                )
            resolved[name] = env_value
            continue
        resolved[name] = str(value)
    return resolved


def validate_training_cloud_env_separation(
    *,
    secrets: Mapping[str, str] | None,
    env: Mapping[str, str] | None,
) -> None:
    if not secrets or not env:
        return
    overlapping = secrets.keys() & env.keys()
    if overlapping:
        raise ValueError(
            "cloud env keys must not overlap with secrets: "
            + ", ".join(sorted(overlapping))
        )


def redact_training_cloud_value(value: Any, *, secret_values: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        redacted = value
        for secret in sorted(set(secret_values), key=len, reverse=True):
            if len(secret) >= TRAINING_CLOUD_CONTROL_SECRET_MIN_REDACTION_CHARS:
                redacted = redacted.replace(
                    secret, TRAINING_CLOUD_CONTROL_REDACTION_PLACEHOLDER
                )
        return redacted
    if isinstance(value, list):
        return [
            redact_training_cloud_value(item, secret_values=secret_values)
            for item in value
        ]
    if isinstance(value, tuple):
        return [
            redact_training_cloud_value(item, secret_values=secret_values)
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: redact_training_cloud_value(item, secret_values=secret_values)
            for key, item in value.items()
        }
    return value


def serialize_training_cloud_payload(
    body: Mapping[str, Any],
    *,
    secret_values: tuple[str, ...] = (),
    max_bytes: int = TRAINING_CLOUD_CONTROL_INLINE_PAYLOAD_MAX_BYTES,
) -> TrainingCloudPayload:
    redacted_body = redact_training_cloud_value(dict(body), secret_values=secret_values)
    payload_bytes = _canonical_payload_bytes(redacted_body)
    size_bytes = len(payload_bytes)
    if size_bytes > max_bytes:
        raise ValueError(
            "Training cloud payload exceeds inline submission limit "
            f"({size_bytes} bytes > {max_bytes} bytes). "
            "Artifact uploads are not implemented yet."
        )
    return TrainingCloudPayload(
        format=TRAINING_CLOUD_CONTROL_PAYLOAD_FORMAT,
        body=redacted_body,
        sha256=hashlib.sha256(payload_bytes).hexdigest(),
        size_bytes=size_bytes,
    )


def build_training_cloud_run_create_request(
    *,
    provider: str,
    name: str,
    body: Mapping[str, Any],
    runtime: TrainingCloudRuntimeConfig,
    manifest: Mapping[str, Any] | None = None,
    secrets: Mapping[str, object | None] | None = None,
    env: Mapping[str, object | None] | None = None,
) -> TrainingCloudRunCreateRequest:
    resolved_secrets = resolve_training_cloud_env_values(secrets)
    resolved_env = resolve_training_cloud_env_values(env)
    validate_training_cloud_env_separation(secrets=resolved_secrets, env=resolved_env)
    secret_values = tuple(resolved_secrets.values()) if resolved_secrets else ()
    return TrainingCloudRunCreateRequest(
        provider=provider,
        name=name,
        runtime=runtime,
        payload=serialize_training_cloud_payload(body, secret_values=secret_values),
        manifest=redact_training_cloud_value(dict(manifest), secret_values=secret_values)
        if manifest is not None
        else None,
        secrets=resolved_secrets,
        env=resolved_env,
    )


def _canonical_payload_bytes(body: Mapping[str, Any]) -> bytes:
    return json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _require_optional_positive(name: str, value: int | None) -> None:
    if value is not None:
        _require_positive(name, value)
