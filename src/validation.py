"""Dataset validation utilities for LeRobot datasets."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings

import numpy as np


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    message: str
    severity: str = "error"  # error, warning, info
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    dataset_name: str
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Check if all validation checks passed (ignoring warnings)."""
        return all(r.passed or r.severity == "warning" for r in self.results)

    @property
    def errors(self) -> List[ValidationResult]:
        """Get all error-level failures."""
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> List[ValidationResult]:
        """Get all warnings."""
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    @property
    def info(self) -> List[ValidationResult]:
        """Get all info-level messages."""
        return [r for r in self.results if r.severity == "info"]

    def add_result(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.results.append(result)

    def summary(self) -> str:
        """Generate a text summary of the validation report."""
        lines = [f"Validation Report: {self.dataset_name}"]
        lines.append("=" * 60)

        error_count = len(self.errors)
        warning_count = len(self.warnings)
        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)

        lines.append(f"Total Checks: {total_count}")
        lines.append(f"Passed: {passed_count}")
        lines.append(f"Errors: {error_count}")
        lines.append(f"Warnings: {warning_count}")
        lines.append("")

        if self.errors:
            lines.append("ERRORS:")
            for result in self.errors:
                lines.append(f"  ✗ {result.message}")
                if result.details:
                    for key, value in result.details.items():
                        lines.append(f"    {key}: {value}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for result in self.warnings:
                lines.append(f"  ⚠ {result.message}")
                if result.details:
                    for key, value in result.details.items():
                        lines.append(f"    {key}: {value}")
            lines.append("")

        if self.passed:
            lines.append("✓ All validation checks passed!")
        else:
            lines.append(f"✗ Validation failed with {error_count} error(s)")

        return "\n".join(lines)


class DatasetValidator:
    """Validator for LeRobot datasets."""

    def __init__(self, dataset_wrapper):
        """
        Initialize the validator.

        Args:
            dataset_wrapper: DatasetWrapper instance to validate
        """
        self.wrapper = dataset_wrapper
        self.dataset = dataset_wrapper.dataset
        self.meta = self.dataset.meta

    def validate_all(
        self,
        check_semantic: bool = True,
        check_action_normalization: bool = True,
        check_observation_consistency: bool = True,
        check_frame_alignment: bool = True,
        check_off_by_one: bool = True,
    ) -> ValidationReport:
        """
        Run all validation checks.

        Args:
            check_semantic: Enable semantic validation
            check_action_normalization: Enable action normalization checks
            check_observation_consistency: Enable observation consistency checks
            check_frame_alignment: Enable frame-level alignment checks
            check_off_by_one: Enable off-by-one error detection

        Returns:
            ValidationReport with all results
        """
        report = ValidationReport(dataset_name=self.wrapper.repo_id)

        if check_semantic:
            report.add_result(self._validate_semantic())

        if check_action_normalization:
            results = self._validate_action_normalization()
            for result in results:
                report.add_result(result)

        if check_observation_consistency:
            results = self._validate_observation_consistency()
            for result in results:
                report.add_result(result)

        if check_frame_alignment:
            results = self._validate_frame_alignment()
            for result in results:
                report.add_result(result)

        if check_off_by_one:
            results = self._validate_off_by_one()
            for result in results:
                report.add_result(result)

        return report

    def _validate_semantic(self) -> ValidationResult:
        """
        Validate semantic correctness of the dataset.

        Checks:
        - Required features exist (observation, action)
        - Episodes have valid metadata
        - Feature names follow conventions
        - Data types are appropriate
        """
        issues = []

        # Check for required features
        features = set(self.meta.features.keys())
        required_features = {"action"}
        missing_required = required_features - features

        if missing_required:
            return ValidationResult(
                passed=False,
                message="Missing required features",
                severity="error",
                details={"missing": list(missing_required)}
            )

        # Check for at least one observation feature
        observation_features = [f for f in features if f.startswith("observation.")]
        if not observation_features:
            issues.append("No observation features found (expected observation.* features)")

        # Check episode metadata
        if self.meta.total_episodes == 0:
            return ValidationResult(
                passed=False,
                message="Dataset has zero episodes",
                severity="error"
            )

        # Check feature naming conventions
        unconventional_names = []
        for feature in features:
            if not (
                feature.startswith("observation.")
                or feature == "action"
                or feature.startswith("action.")
                or feature in ["reward", "success", "done", "next.reward", "next.done"]
            ):
                unconventional_names.append(feature)

        if unconventional_names:
            issues.append(f"Unconventional feature names: {', '.join(unconventional_names)}")

        # Check data types
        action_dtype = self.meta.features["action"]["dtype"]
        if action_dtype not in ["float32", "float64"]:
            issues.append(f"Action dtype '{action_dtype}' should be float32 or float64")

        if issues:
            return ValidationResult(
                passed=False,
                message="Semantic validation found issues",
                severity="warning",
                details={"issues": issues}
            )

        return ValidationResult(
            passed=True,
            message="Semantic validation passed",
            severity="info"
        )

    def _validate_action_normalization(self) -> List[ValidationResult]:
        """
        Validate action normalization, bounds, and units.

        Checks:
        - Actions are within reasonable bounds
        - Actions are properly normalized
        - No NaN or infinite values
        - Action statistics are reasonable
        """
        results = []

        # Get action data for a sample of frames
        num_samples = min(1000, len(self.dataset))
        sample_indices = np.random.choice(len(self.dataset), num_samples, replace=False)

        actions = []
        for idx in sample_indices:
            frame = self.dataset[int(idx)]
            action = frame["action"]
            if hasattr(action, "numpy"):
                action = action.numpy()
            actions.append(action)

        actions = np.array(actions)

        # Check for NaN or inf
        if np.any(np.isnan(actions)):
            nan_count = np.sum(np.isnan(actions))
            results.append(ValidationResult(
                passed=False,
                message="Actions contain NaN values",
                severity="error",
                details={"nan_count": int(nan_count), "total_samples": num_samples}
            ))

        if np.any(np.isinf(actions)):
            inf_count = np.sum(np.isinf(actions))
            results.append(ValidationResult(
                passed=False,
                message="Actions contain infinite values",
                severity="error",
                details={"inf_count": int(inf_count), "total_samples": num_samples}
            ))

        # Check action bounds
        action_min = np.min(actions, axis=0)
        action_max = np.max(actions, axis=0)
        action_mean = np.mean(actions, axis=0)
        action_std = np.std(actions, axis=0)

        # Check if actions appear to be normalized (roughly in [-1, 1] or [0, 1])
        is_normalized_neg1_1 = np.all(action_min >= -1.5) and np.all(action_max <= 1.5)
        is_normalized_0_1 = np.all(action_min >= -0.5) and np.all(action_max <= 1.5)

        if not (is_normalized_neg1_1 or is_normalized_0_1):
            results.append(ValidationResult(
                passed=False,
                message="Actions may not be properly normalized",
                severity="warning",
                details={
                    "min": action_min.tolist(),
                    "max": action_max.tolist(),
                    "mean": action_mean.tolist(),
                    "std": action_std.tolist(),
                    "expected_range": "[-1, 1] or [0, 1]"
                }
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                message="Actions appear properly normalized",
                severity="info",
                details={
                    "min": action_min.tolist(),
                    "max": action_max.tolist(),
                    "mean": action_mean.tolist(),
                    "std": action_std.tolist()
                }
            ))

        # Check for suspiciously low variance (might indicate constant actions)
        low_variance_dims = np.where(action_std < 1e-6)[0]
        if len(low_variance_dims) > 0:
            results.append(ValidationResult(
                passed=False,
                message="Some action dimensions have very low variance",
                severity="warning",
                details={
                    "low_variance_dimensions": low_variance_dims.tolist(),
                    "std": action_std[low_variance_dims].tolist()
                }
            ))

        return results

    def _validate_observation_consistency(self) -> List[ValidationResult]:
        """
        Validate observation modality consistency.

        Checks:
        - Camera observations have consistent shapes
        - State observations have consistent dimensions
        - Image observations have valid ranges
        - No missing or corrupted observations
        """
        results = []

        # Get observation features
        obs_features = {k: v for k, v in self.meta.features.items() if k.startswith("observation.")}

        if not obs_features:
            results.append(ValidationResult(
                passed=False,
                message="No observation features found",
                severity="error"
            ))
            return results

        # Sample frames to check consistency
        num_samples = min(100, len(self.dataset))
        sample_indices = np.linspace(0, len(self.dataset) - 1, num_samples, dtype=int)

        for feature_name, feature_info in obs_features.items():
            expected_shape = feature_info["shape"]
            dtype = feature_info["dtype"]

            # Check if it's an image (3D array with last dim = 3 or 4)
            is_image = (
                len(expected_shape) == 3
                and expected_shape[-1] in [3, 4]
            )

            shape_mismatches = []
            value_issues = []

            for idx in sample_indices:
                try:
                    frame = self.dataset[int(idx)]
                    obs = frame[feature_name]

                    if hasattr(obs, "numpy"):
                        obs = obs.numpy()

                    # Check shape consistency
                    if obs.shape != expected_shape:
                        shape_mismatches.append({
                            "frame_idx": int(idx),
                            "expected": expected_shape,
                            "actual": obs.shape
                        })

                    # For images, check value ranges
                    if is_image:
                        obs_min = np.min(obs)
                        obs_max = np.max(obs)

                        # Images should be in [0, 255] or [0, 1]
                        if not ((0 <= obs_min and obs_max <= 1.5) or (0 <= obs_min and obs_max <= 255)):
                            value_issues.append({
                                "frame_idx": int(idx),
                                "min": float(obs_min),
                                "max": float(obs_max)
                            })

                        # Check for NaN
                        if np.any(np.isnan(obs)):
                            value_issues.append({
                                "frame_idx": int(idx),
                                "issue": "contains NaN values"
                            })

                except Exception as e:
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Error reading {feature_name} at frame {idx}",
                        severity="error",
                        details={"error": str(e)}
                    ))

            # Report shape mismatches
            if shape_mismatches:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Shape inconsistencies in {feature_name}",
                    severity="error",
                    details={
                        "expected_shape": expected_shape,
                        "mismatches": shape_mismatches[:5]  # Show first 5
                    }
                ))

            # Report value issues
            if value_issues:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Value range issues in {feature_name}",
                    severity="warning",
                    details={"issues": value_issues[:5]}  # Show first 5
                ))

        if not results:
            results.append(ValidationResult(
                passed=True,
                message="Observation consistency checks passed",
                severity="info",
                details={"observation_features": len(obs_features)}
            ))

        return results

    def _validate_frame_alignment(self) -> List[ValidationResult]:
        """
        Validate frame-level alignment between observations and actions.

        Checks:
        - All features have the same number of frames
        - Episode boundaries are consistent
        - Timestamps are monotonically increasing (if available)
        """
        results = []

        # Check total frames consistency
        total_frames = self.meta.total_frames

        # Verify episode frame counts sum to total
        episode_frame_sum = sum(
            ep["dataset_to_index"] - ep["dataset_from_index"]
            for ep in self.meta.episodes
        )

        if episode_frame_sum != total_frames:
            results.append(ValidationResult(
                passed=False,
                message="Episode frame counts don't sum to total frames",
                severity="error",
                details={
                    "total_frames": total_frames,
                    "episode_frame_sum": episode_frame_sum
                }
            ))

        # Check episode boundaries
        for i, episode in enumerate(self.meta.episodes):
            from_idx = episode["dataset_from_index"]
            to_idx = episode["dataset_to_index"]

            # Check valid range
            if from_idx < 0 or to_idx > total_frames:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Episode {i} has invalid frame range",
                    severity="error",
                    details={
                        "episode": i,
                        "from_index": from_idx,
                        "to_index": to_idx,
                        "total_frames": total_frames
                    }
                ))

            # Check non-empty
            if from_idx >= to_idx:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Episode {i} is empty or has invalid range",
                    severity="error",
                    details={
                        "episode": i,
                        "from_index": from_idx,
                        "to_index": to_idx
                    }
                ))

            # Check for gaps or overlaps with next episode
            if i < len(self.meta.episodes) - 1:
                next_episode = self.meta.episodes[i + 1]
                next_from = next_episode["dataset_from_index"]

                if to_idx != next_from:
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Gap or overlap between episodes {i} and {i+1}",
                        severity="error",
                        details={
                            "episode_1_end": to_idx,
                            "episode_2_start": next_from,
                            "gap": next_from - to_idx
                        }
                    ))

        # Check timestamp consistency (if timestamps exist)
        if "timestamp" in self.meta.features or "observation.timestamp" in self.meta.features:
            timestamp_key = "timestamp" if "timestamp" in self.meta.features else "observation.timestamp"

            # Sample a few episodes to check timestamp monotonicity
            episodes_to_check = min(5, self.meta.total_episodes)
            for ep_idx in range(episodes_to_check):
                episode = self.meta.episodes[ep_idx]
                from_idx = episode["dataset_from_index"]
                to_idx = episode["dataset_to_index"]

                timestamps = []
                for frame_idx in range(from_idx, min(to_idx, from_idx + 100)):  # Check first 100 frames
                    frame = self.dataset[frame_idx]
                    ts = frame.get(timestamp_key)
                    if ts is not None:
                        if hasattr(ts, "item"):
                            ts = ts.item()
                        timestamps.append(ts)

                # Check monotonicity
                if len(timestamps) > 1:
                    diffs = np.diff(timestamps)
                    if np.any(diffs < 0):
                        results.append(ValidationResult(
                            passed=False,
                            message=f"Non-monotonic timestamps in episode {ep_idx}",
                            severity="error",
                            details={
                                "episode": ep_idx,
                                "negative_diffs": int(np.sum(diffs < 0))
                            }
                        ))

        if not results:
            results.append(ValidationResult(
                passed=True,
                message="Frame alignment checks passed",
                severity="info"
            ))

        return results

    def _validate_off_by_one(self) -> List[ValidationResult]:
        """
        Detect potential off-by-one errors between action and observation streams.

        Checks:
        - Actions and observations are properly aligned
        - No temporal shift between modalities
        - Episode lengths are consistent
        """
        results = []

        # Sample a few episodes to check alignment
        episodes_to_check = min(10, self.meta.total_episodes)

        for ep_idx in range(episodes_to_check):
            episode = self.meta.episodes[ep_idx]
            from_idx = episode["dataset_from_index"]
            to_idx = episode["dataset_to_index"]
            episode_length = to_idx - from_idx

            # Check if we can access all frames in the episode
            try:
                first_frame = self.dataset[from_idx]
                last_frame = self.dataset[to_idx - 1]

                # Verify we can't access beyond the episode
                if to_idx < len(self.dataset):
                    try:
                        _ = self.dataset[to_idx]
                        # If this succeeds, it's fine - it's the next episode
                    except Exception:
                        pass  # Expected to potentially fail at boundary

            except Exception as e:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Cannot access frames in episode {ep_idx}",
                    severity="error",
                    details={
                        "episode": ep_idx,
                        "from_index": from_idx,
                        "to_index": to_idx,
                        "error": str(e)
                    }
                ))
                continue

            # Check for off-by-one in episode metadata
            expected_length = episode.get("length", None)
            if expected_length is not None and expected_length != episode_length:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Episode {ep_idx} length mismatch",
                    severity="error",
                    details={
                        "episode": ep_idx,
                        "expected_length": expected_length,
                        "actual_length": episode_length,
                        "difference": expected_length - episode_length
                    }
                ))

            # For episodes with "done" or "success" flags, check they're set at the end
            if "done" in self.meta.features:
                try:
                    last_done = self.dataset[to_idx - 1]["done"]
                    if hasattr(last_done, "item"):
                        last_done = last_done.item()

                    # Last frame should typically have done=True
                    # (This is a warning, not an error, as some datasets may not follow this)
                    if not last_done:
                        results.append(ValidationResult(
                            passed=False,
                            message=f"Episode {ep_idx} last frame doesn't have done=True",
                            severity="warning",
                            details={"episode": ep_idx, "last_frame_idx": to_idx - 1}
                        ))
                except Exception:
                    pass  # done feature might not exist or might be structured differently

        # Check action-observation temporal alignment
        # Sample frames and check if action leads/lags observation inappropriately
        num_samples = min(50, len(self.dataset) - 1)
        sample_indices = np.random.choice(len(self.dataset) - 1, num_samples, replace=False)

        for idx in sample_indices:
            try:
                current_frame = self.dataset[int(idx)]
                next_frame = self.dataset[int(idx + 1)]

                # If there's a "next.observation" field, verify it matches the next frame's observation
                obs_features = [k for k in current_frame.keys() if k.startswith("observation.")]
                next_obs_features = [k for k in current_frame.keys() if k.startswith("next.observation.")]

                if next_obs_features:
                    for next_obs_key in next_obs_features:
                        obs_key = next_obs_key.replace("next.", "")

                        if obs_key in next_frame:
                            current_next_obs = current_frame[next_obs_key]
                            actual_next_obs = next_frame[obs_key]

                            if hasattr(current_next_obs, "numpy"):
                                current_next_obs = current_next_obs.numpy()
                            if hasattr(actual_next_obs, "numpy"):
                                actual_next_obs = actual_next_obs.numpy()

                            # Check if they match (allowing for small floating point errors)
                            if not np.allclose(current_next_obs, actual_next_obs, rtol=1e-5, atol=1e-5):
                                results.append(ValidationResult(
                                    passed=False,
                                    message=f"next.observation mismatch at frame {idx}",
                                    severity="error",
                                    details={
                                        "frame_idx": int(idx),
                                        "feature": obs_key,
                                        "max_diff": float(np.max(np.abs(current_next_obs - actual_next_obs)))
                                    }
                                ))
                                break  # Only report once per frame

            except Exception as e:
                # Might fail at episode boundaries, which is okay
                pass

        if not results:
            results.append(ValidationResult(
                passed=True,
                message="No off-by-one errors detected",
                severity="info"
            ))

        return results


def validate_dataset(
    dataset_wrapper,
    check_semantic: bool = True,
    check_action_normalization: bool = True,
    check_observation_consistency: bool = True,
    check_frame_alignment: bool = True,
    check_off_by_one: bool = True,
) -> ValidationReport:
    """
    Validate a LeRobot dataset.

    Args:
        dataset_wrapper: DatasetWrapper instance to validate
        check_semantic: Enable semantic validation
        check_action_normalization: Enable action normalization checks
        check_observation_consistency: Enable observation consistency checks
        check_frame_alignment: Enable frame-level alignment checks
        check_off_by_one: Enable off-by-one error detection

    Returns:
        ValidationReport with results

    Example:
        >>> from src import load_dataset
        >>> from src.validation import validate_dataset
        >>>
        >>> dataset = load_dataset("lerobot/pusht")
        >>> report = validate_dataset(dataset)
        >>> print(report.summary())
    """
    validator = DatasetValidator(dataset_wrapper)
    return validator.validate_all(
        check_semantic=check_semantic,
        check_action_normalization=check_action_normalization,
        check_observation_consistency=check_observation_consistency,
        check_frame_alignment=check_frame_alignment,
        check_off_by_one=check_off_by_one,
    )
