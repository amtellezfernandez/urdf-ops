from __future__ import annotations

from backend.models.datasets import (
    DatasetMixExecutionPlan,
    DatasetMixPartitionPlan,
    DatasetMixSourceRef,
)
from backend.services.dataset_mix_lerobot import can_execute_native_local_lerobot_sources
from backend.services.dataset_mix_lerobot_params import (
    DATASET_MIX_LEROBOT_PARTITION_EPISODES_PER_PARTITION,
    DATASET_MIX_LEROBOT_PARTITION_FRAMES_PER_PARTITION,
)

DATASET_MIX_EXECUTION_REASON_NATIVE_LOCAL = (
    "All sources are local LeRobot datasets. The worker will prefer official "
    "LeRobot dataset tools and only use the local compatibility runtime when "
    "official LeRobot is unavailable."
)
DATASET_MIX_EXECUTION_REASON_LEGACY_FALLBACK = (
    "At least one source requires the legacy subprocess path because the local "
    "LeRobot execution path does not support this source set."
)


def compile_dataset_mix_execution_plan(
    *,
    sources: list[DatasetMixSourceRef],
) -> DatasetMixExecutionPlan:
    if can_execute_native_local_lerobot_for_sources(sources):
        return DatasetMixExecutionPlan(
            execution_mode="native-local-lerobot",
            reason=DATASET_MIX_EXECUTION_REASON_NATIVE_LOCAL,
        )
    return DatasetMixExecutionPlan(
        execution_mode="legacy-subprocess",
        reason=DATASET_MIX_EXECUTION_REASON_LEGACY_FALLBACK,
    )


def can_execute_native_local_lerobot_for_sources(
    sources: list[DatasetMixSourceRef],
) -> bool:
    return can_execute_native_local_lerobot_sources(sources)


def compile_dataset_mix_partition_plan(
    *,
    execution_plan: DatasetMixExecutionPlan,
) -> DatasetMixPartitionPlan | None:
    if execution_plan.execution_mode != "native-local-lerobot":
        return None
    return DatasetMixPartitionPlan(
        strategy="episode-window",
        target_episodes_per_partition=DATASET_MIX_LEROBOT_PARTITION_EPISODES_PER_PARTITION,
        target_frames_per_partition=DATASET_MIX_LEROBOT_PARTITION_FRAMES_PER_PARTITION,
    )
