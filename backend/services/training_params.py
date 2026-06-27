from __future__ import annotations

from typing import Final


TRAINING_REQUIRED_RUNTIME_MODULES: Final[tuple[str, ...]] = ("torch", "lerobot")
TRAINING_OPTIONAL_RUNTIME_MODULES: Final[tuple[str, ...]] = ("safetensors",)
TRAINING_LEROBOT_PYTHON_ENV: Final = "URDF_STUDIO_TRAINING_LEROBOT_PYTHON"
TRAINING_LEROBOT_TOOLCHAIN_DIRNAME: Final = ".venv-lerobot"
TRAINING_LOCAL_COMPUTE_TYPE: Final = "local"
TRAINING_BYOC_COMPUTE_TYPES: Final[tuple[str, ...]] = ("ssh",)
TRAINING_DEFAULT_OUTPUT_DIR: Final = "./outputs"
TRAINING_DEFAULT_OUTPUT_ROOT_DIRNAME: Final = "outputs"
TRAINING_OUTPUT_ROOTS_ENV: Final = "URDF_STUDIO_TRAINING_OUTPUT_ROOTS"
TRAINING_CLOUD_COMPUTE_TYPES: Final[tuple[str, ...]] = ("modal", "runpod", "macrodata")
TRAINING_PLANNED_COMPUTE_TYPES: Final[tuple[str, ...]] = ("aws",)
TRAINING_CLOUD_CONTROL_REQUIRED_CAPABILITIES: Final[tuple[str, ...]] = (
    "provider_submit",
    "status_polling",
    "log_streaming",
    "remote_cancel",
    "artifact_download",
)
TRAINING_COMPUTE_BACKEND_LABELS: Final[dict[str, str]] = {
    "local": "Local GPU",
    "ssh": "Remote Docker machine",
    "modal": "Modal",
    "runpod": "RunPod",
    "macrodata": "Macrodata Cloud",
    "aws": "AWS",
}
TRAINING_CLOUD_COMPUTE_DISABLED_MESSAGE: Final = (
    "Cloud training runners are disabled in this build. Use local training until "
    "provider execution, log streaming, artifact download, and cancellation are wired."
)
