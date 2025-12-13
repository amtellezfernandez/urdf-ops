"""LeRobot Dataset Wrapper - Reference implementation for dataset operations."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.streaming_dataset import StreamingLeRobotDataset
from lerobot.datasets.dataset_tools import (
    delete_episodes as _delete_episodes,
    split_dataset as _split_dataset,
    merge_datasets as _merge_datasets,
    add_features as _add_features,
    remove_feature as _remove_feature,
    modify_features as _modify_features,
)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "datasets"


class DatasetWrapper:
    """Wrapper for LeRobot datasets providing high-level operations."""

    def __init__(
        self,
        repo_id: str,
        root: Optional[Union[str, Path]] = None,
        episodes: Optional[List[int]] = None,
        image_transforms: Optional[Callable] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        stream: bool = False,
    ):
        """
        Initialize the DatasetWrapper.

        Args:
            repo_id: HuggingFace repository ID (e.g., "lerobot/pusht")
            root: Local root directory for the dataset
            episodes: Optional list of episode indices to load
            image_transforms: Optional image transformation function
            delta_timestamps: Optional delta timestamps configuration
            tolerance_s: Tolerance for timestamp matching
            download_videos: Whether to download video files
            stream: If True, use StreamingLeRobotDataset to stream from Hub without local copies
        """
        self.repo_id = repo_id
        self.root = Path(root) if root else None
        self.stream = stream

        if stream:
            # Streaming datasets stream directly from the Hub without local copies
            # Note: Some parameters may not be supported for streaming datasets
            self.dataset = StreamingLeRobotDataset(repo_id)
        else:
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                episodes=episodes,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=tolerance_s,
                download_videos=download_videos,
            )

    def __repr__(self) -> str:
        mode = "streaming" if self.stream else "local"
        return (
            f"DatasetWrapper(repo_id='{self.repo_id}', "
            f"mode={mode}, "
            f"episodes={self.dataset.meta.total_episodes}, "
            f"frames={self.dataset.meta.total_frames})"
        )

    def __len__(self) -> int:
        """Return the total number of frames in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a frame by index."""
        return self.dataset[idx]

    def info(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset information.

        Returns:
            Dictionary containing dataset metadata, features, and statistics
        """
        meta = self.dataset.meta
        return {
            "repo_id": self.repo_id,
            "total_episodes": meta.total_episodes,
            "total_frames": meta.total_frames,
            "fps": meta.fps,
            "robot_type": meta.robot_type,
            "features": list(meta.features.keys()),
            "camera_keys": list(meta.camera_keys),
            "video_keys": list(meta.video_keys),
            "shapes": meta.shapes,
            "dtypes": {k: v["dtype"] for k, v in meta.features.items()},
        }

    def get_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Get all frames from a specific episode.

        Args:
            episode_idx: Episode index to retrieve

        Returns:
            Dictionary containing all episode frames and metadata
        """
        if episode_idx >= self.dataset.meta.total_episodes:
            raise ValueError(
                f"Episode index {episode_idx} out of range "
                f"(dataset has {self.dataset.meta.total_episodes} episodes)"
            )

        episode_meta = self.dataset.meta.episodes[episode_idx]
        start_idx = episode_meta["dataset_from_index"]
        end_idx = episode_meta["dataset_to_index"]

        frames = []
        for idx in range(start_idx, end_idx):
            frames.append(self.dataset[idx])

        return {
            "episode_index": episode_idx,
            "frames": frames,
            "metadata": dict(episode_meta),
            "length": end_idx - start_idx,
        }

    def get_episode_info(self, episode_idx: int) -> Dict[str, Any]:
        """
        Get metadata about a specific episode without loading frames.

        Args:
            episode_idx: Episode index

        Returns:
            Dictionary containing episode metadata
        """
        if episode_idx >= self.dataset.meta.total_episodes:
            raise ValueError(
                f"Episode index {episode_idx} out of range "
                f"(dataset has {self.dataset.meta.total_episodes} episodes)"
            )

        episode_meta = self.dataset.meta.episodes[episode_idx]
        return dict(episode_meta)

    def get_episodes(self, episode_indices: List[int]) -> List[Dict[str, Any]]:
        """
        Get multiple episodes at once.

        Args:
            episode_indices: List of episode indices to retrieve

        Returns:
            List of episode dictionaries
        """
        return [self.get_episode(idx) for idx in episode_indices]

    def get_frame(self, frame_idx: int) -> Dict[str, Any]:
        """
        Get a single frame by global frame index.

        Args:
            frame_idx: Global frame index

        Returns:
            Frame data dictionary
        """
        return self.dataset[frame_idx]

    def get_frames(self, start_idx: int, end_idx: int) -> List[Dict[str, Any]]:
        """
        Get a range of frames.

        Args:
            start_idx: Starting frame index
            end_idx: Ending frame index (exclusive)

        Returns:
            List of frame dictionaries
        """
        return [self.dataset[idx] for idx in range(start_idx, end_idx)]

    def delete_episodes(
        self,
        episode_indices: List[int],
        new_repo_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "DatasetWrapper":
        """
        Delete specific episodes and create a new dataset.

        Args:
            episode_indices: List of episode indices to delete
            new_repo_id: Repository ID for the new dataset
            output_dir: Directory to save the new dataset (defaults to datasets/)

        Returns:
            New DatasetWrapper instance with episodes removed
        """
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR

        new_dataset = _delete_episodes(
            self.dataset,
            episode_indices=episode_indices,
            repo_id=new_repo_id,
            output_dir=output_dir,
        )

        return DatasetWrapper(
            repo_id=new_dataset.repo_id,
            root=new_dataset.root,
        )

    def split(
        self,
        splits: Dict[str, Union[float, List[int]]],
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, "DatasetWrapper"]:
        """
        Split the dataset into multiple subsets.

        Args:
            splits: Dictionary defining the splits. Can be:
                - Fractions: {"train": 0.8, "val": 0.2}
                - Episode indices: {"train": [0, 1, 2], "val": [3, 4]}
            output_dir: Directory to save split datasets (defaults to datasets/)

        Returns:
            Dictionary mapping split names to DatasetWrapper instances
        """
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR

        split_datasets = _split_dataset(
            self.dataset,
            splits=splits,
            output_dir=output_dir,
        )

        return {
            name: DatasetWrapper(repo_id=dataset.repo_id, root=dataset.root)
            for name, dataset in split_datasets.items()
        }

    def add_features(
        self,
        features: Dict[str, tuple],
        new_repo_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "DatasetWrapper":
        """
        Add new features to the dataset.

        Args:
            features: Dictionary mapping feature names to (data, schema) tuples.
                data can be a numpy array or a callable function.
                schema is a dict with keys: dtype, shape, names
            new_repo_id: Repository ID for the new dataset
            output_dir: Directory to save the new dataset (defaults to datasets/)

        Returns:
            New DatasetWrapper instance with added features

        """
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR

        new_dataset = _add_features(
            self.dataset,
            features=features,
            repo_id=new_repo_id,
            output_dir=output_dir,
        )

        return DatasetWrapper(
            repo_id=new_dataset.repo_id,
            root=new_dataset.root,
        )

    def remove_features(
        self,
        feature_names: Union[str, List[str]],
        new_repo_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "DatasetWrapper":
        """
        Remove features from the dataset.

        Args:
            feature_names: Feature name or list of feature names to remove
            new_repo_id: Repository ID for the new dataset
            output_dir: Directory to save the new dataset (defaults to datasets/)

        Returns:
            New DatasetWrapper instance with features removed
        """
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR

        new_dataset = _remove_feature(
            self.dataset,
            feature_names=feature_names,
            repo_id=new_repo_id,
            output_dir=output_dir,
        )

        return DatasetWrapper(
            repo_id=new_dataset.repo_id,
            root=new_dataset.root,
        )

    def modify_features(
        self,
        add_features: Optional[Dict[str, tuple]] = None,
        remove_features: Optional[Union[str, List[str]]] = None,
        new_repo_id: Optional[str] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> "DatasetWrapper":
        """
        Add and remove features in a single operation.

        Args:
            add_features: Dictionary of features to add
            remove_features: Feature name(s) to remove
            new_repo_id: Repository ID for the new dataset
            output_dir: Directory to save the new dataset (defaults to datasets/)

        Returns:
            New DatasetWrapper instance with modified features
        """
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR

        new_dataset = _modify_features(
            self.dataset,
            add_features=add_features or {},
            remove_features=remove_features,
            repo_id=new_repo_id,
            output_dir=output_dir,
        )

        return DatasetWrapper(
            repo_id=new_dataset.repo_id,
            root=new_dataset.root,
        )

    def to_torch_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        **kwargs,
    ) -> torch.utils.data.DataLoader:
        """
        Create a PyTorch DataLoader from the dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes
            **kwargs: Additional arguments passed to DataLoader

        Returns:
            PyTorch DataLoader instance
        """
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )

    def validate(
        self,
        check_semantic: bool = True,
        check_action_normalization: bool = True,
        check_observation_consistency: bool = True,
        check_frame_alignment: bool = True,
        check_off_by_one: bool = True,
    ):
        """
        Validate the dataset for common issues.

        Args:
            check_semantic: Enable semantic validation
            check_action_normalization: Enable action normalization checks
            check_observation_consistency: Enable observation consistency checks
            check_frame_alignment: Enable frame-level alignment checks
            check_off_by_one: Enable off-by-one error detection

        Returns:
            ValidationReport with results

        Example:
            >>> dataset = load_dataset("lerobot/pusht")
            >>> report = dataset.validate()
            >>> print(report.summary())
        """
        from validation import validate_dataset

        return validate_dataset(
            self,
            check_semantic=check_semantic,
            check_action_normalization=check_action_normalization,
            check_observation_consistency=check_observation_consistency,
            check_frame_alignment=check_frame_alignment,
            check_off_by_one=check_off_by_one,
        )


# Convenience functions

def load_dataset(
    repo_id: str,
    root: Optional[Union[str, Path]] = None,
    episodes: Optional[List[int]] = None,
    stream: bool = False,
    **kwargs,
) -> DatasetWrapper:
    """
    Load a LeRobot dataset with a convenient wrapper.

    Args:
        repo_id: HuggingFace repository ID (e.g., "lerobot/pusht")
        root: Local root directory for the dataset
        episodes: Optional list of episode indices to load
        stream: If True, stream dataset directly from Hub without local copies
        **kwargs: Additional arguments passed to DatasetWrapper

    Returns:
        DatasetWrapper instance

    """
    return DatasetWrapper(repo_id=repo_id, root=root, episodes=episodes, stream=stream, **kwargs)


def load_episode(
    repo_id: str,
    episode_idx: int,
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Load a single episode from a dataset.

    Args:
        repo_id: HuggingFace repository ID
        episode_idx: Episode index to load
        root: Local root directory for the dataset

    Returns:
        Dictionary containing episode frames and metadata

    """
    dataset = load_dataset(repo_id, root=root, episodes=[episode_idx])
    return dataset.get_episode(episode_idx)


def load_episodes(
    repo_id: str,
    episode_indices: List[int],
    root: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Load multiple episodes from a dataset.

    Args:
        repo_id: HuggingFace repository ID
        episode_indices: List of episode indices to load
        root: Local root directory for the dataset

    Returns:
        List of episode dictionaries

    """
    dataset = load_dataset(repo_id, root=root, episodes=episode_indices)
    return dataset.get_episodes(episode_indices)


def get_episode_info(
    repo_id: str,
    episode_idx: int,
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Get metadata about a specific episode without loading frames.

    Args:
        repo_id: HuggingFace repository ID
        episode_idx: Episode index
        root: Local root directory for the dataset

    Returns:
        Dictionary containing episode metadata

    """
    dataset = load_dataset(repo_id, root=root)
    return dataset.get_episode_info(episode_idx)


def get_dataset_info(
    repo_id: str,
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Get comprehensive information about a dataset.

    Args:
        repo_id: HuggingFace repository ID
        root: Local root directory for the dataset

    Returns:
        Dictionary containing dataset metadata and statistics

    """
    dataset = load_dataset(repo_id, root=root)
    return dataset.info()


def delete_episodes_from_dataset(
    repo_id: str,
    episode_indices: List[int],
    new_repo_id: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
) -> DatasetWrapper:
    """
    Delete episodes from a dataset and create a new one.

    Args:
        repo_id: Source HuggingFace repository ID
        episode_indices: List of episode indices to delete
        new_repo_id: Repository ID for the new dataset
        output_dir: Directory to save the new dataset (defaults to datasets/)
        root: Local root directory for the source dataset

    Returns:
        DatasetWrapper instance for the new dataset

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    dataset = load_dataset(repo_id, root=root)
    return dataset.delete_episodes(episode_indices, new_repo_id, output_dir)


def split_dataset_by_episodes(
    repo_id: str,
    splits: Dict[str, List[int]],
    output_dir: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, DatasetWrapper]:
    """
    Split a dataset by explicit episode indices.

    Args:
        repo_id: Source HuggingFace repository ID
        splits: Dictionary mapping split names to episode indices
        output_dir: Directory to save split datasets (defaults to datasets/)
        root: Local root directory for the source dataset

    Returns:
        Dictionary mapping split names to DatasetWrapper instances

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    dataset = load_dataset(repo_id, root=root)
    return dataset.split(splits, output_dir)


def split_dataset_by_ratio(
    repo_id: str,
    splits: Dict[str, float],
    output_dir: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
) -> Dict[str, DatasetWrapper]:
    """
    Split a dataset by fractional ratios.

    Args:
        repo_id: Source HuggingFace repository ID
        splits: Dictionary mapping split names to fractions (e.g., {"train": 0.8, "val": 0.2})
        output_dir: Directory to save split datasets (defaults to datasets/)
        root: Local root directory for the source dataset

    Returns:
        Dictionary mapping split names to DatasetWrapper instances

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    dataset = load_dataset(repo_id, root=root)
    return dataset.split(splits, output_dir)


def merge_datasets(
    repo_ids: List[str],
    output_repo_id: str,
    output_dir: Optional[Union[str, Path]] = None,
    roots: Optional[List[Union[str, Path]]] = None,
) -> DatasetWrapper:
    """
    Merge multiple datasets into one.

    Note: Datasets must have identical features. Episodes are concatenated
    in the order specified in repo_ids.

    Args:
        repo_ids: List of HuggingFace repository IDs to merge
        output_repo_id: Repository ID for the merged dataset
        output_dir: Directory to save the merged dataset (defaults to datasets/)
        roots: Optional list of local root directories for source datasets

    Returns:
        DatasetWrapper instance for the merged dataset

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR

    if roots is None:
        roots = [None] * len(repo_ids)

    datasets = [
        LeRobotDataset(repo_id=repo_id, root=root)
        for repo_id, root in zip(repo_ids, roots)
    ]

    merged_dataset = _merge_datasets(
        datasets,
        output_repo_id=output_repo_id,
        output_dir=output_dir,
    )

    return DatasetWrapper(
        repo_id=merged_dataset.repo_id,
        root=merged_dataset.root,
    )


def add_features_to_dataset(
    repo_id: str,
    features: Dict[str, tuple],
    new_repo_id: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
) -> DatasetWrapper:
    """
    Add new features to a dataset.

    Args:
        repo_id: Source HuggingFace repository ID
        features: Dictionary mapping feature names to (data, schema) tuples
        new_repo_id: Repository ID for the new dataset
        output_dir: Directory to save the new dataset (defaults to datasets/)
        root: Local root directory for the source dataset

    Returns:
        DatasetWrapper instance for the new dataset

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    dataset = load_dataset(repo_id, root=root)
    return dataset.add_features(features, new_repo_id, output_dir)


def remove_features_from_dataset(
    repo_id: str,
    feature_names: Union[str, List[str]],
    new_repo_id: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    root: Optional[Union[str, Path]] = None,
) -> DatasetWrapper:
    """
    Remove features from a dataset.

    Args:
        repo_id: Source HuggingFace repository ID
        feature_names: Feature name or list of feature names to remove
        new_repo_id: Repository ID for the new dataset
        output_dir: Directory to save the new dataset (defaults to datasets/)
        root: Local root directory for the source dataset

    Returns:
        DatasetWrapper instance for the new dataset

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    dataset = load_dataset(repo_id, root=root)
    return dataset.remove_features(feature_names, new_repo_id, output_dir)
