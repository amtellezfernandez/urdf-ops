from __future__ import annotations

import inspect
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

POLICY_ID_ALIASES = {
    "act": "act",
    "diffusion": "diffusion",
    "diffusion_policy": "diffusion",
    "tdmpc": "tdmpc",
    "vqbet": "vqbet",
    "vq_bet": "vqbet",
}

POLICY_OVERRIDE_ALIASES = {
    "hidden_dim": "dim_model",
    "n_diffusion_steps": "num_train_timesteps",
    "noise_scheduler": "noise_scheduler_type",
}


def normalize_policy_id(architecture: str) -> str:
    """Normalize UI policy IDs to the identifiers used by LeRobot imports."""
    policy_id = str(architecture).strip().lower().replace("-", "_")
    normalized = POLICY_ID_ALIASES.get(policy_id)
    if not normalized:
        raise ValueError(f"Unknown architecture: {architecture}")
    return normalized


def prepare_policy_overrides(policy_config_class, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Filter UI/provider policy overrides against the installed LeRobot config."""
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("Policy model config must be an object.")

    normalized = dict(overrides)
    try:
        signature = inspect.signature(policy_config_class)
    except (TypeError, ValueError):
        logger.warning("Could not inspect %s policy config; passing overrides unchanged.", policy_config_class)
        return normalized

    accepted_keys = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    accepts_arbitrary_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    for old_key, new_key in POLICY_OVERRIDE_ALIASES.items():
        if old_key in normalized and new_key in accepted_keys and new_key not in normalized:
            normalized[new_key] = normalized[old_key]

    if accepts_arbitrary_kwargs:
        return normalized

    filtered = {key: value for key, value in normalized.items() if key in accepted_keys}
    ignored = sorted(set(normalized) - set(filtered))
    if ignored:
        logger.warning(
            "Ignoring unsupported policy config keys for %s: %s",
            getattr(policy_config_class, "__name__", policy_config_class),
            ", ".join(ignored),
        )
    return filtered
