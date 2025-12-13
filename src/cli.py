#!/usr/bin/env python
"""LeRobot Dataset Tool - CLI Interface for dataset manipulation."""

import argparse
import sys
import warnings
import subprocess
import shutil
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

from dataset_wrapper import (
    load_dataset,
    delete_episodes_from_dataset,
    split_dataset_by_ratio,
    add_features_to_dataset,
    merge_datasets as merge_datasets_wrapper,
)
from utils import suppress_output
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


class Colors:
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.RESET} {msg}")


def print_error(msg):
    print(f"{Colors.RED}✗{Colors.RESET} {msg}")


def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.RESET} {msg}")


def print_header(msg):
    print(f"\n{Colors.BOLD}{msg}{Colors.RESET}")


def progress_bar(desc, total=None):
    """Create progress bar if tqdm available, otherwise return None."""
    if HAS_TQDM and total:
        return tqdm(total=total, desc=desc, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
    elif HAS_TQDM:
        return tqdm(desc=desc, bar_format='{desc}: {elapsed}')
    return None


def cmd_info(args):
    print_header(f"Dataset: {args.dataset}")

    try:
        if args.stream:
            print_info(f"Streaming from: {Colors.BLUE}HuggingFace Hub (no local download){Colors.RESET}")
            dataset = load_dataset(args.dataset, stream=True)
        else:
            local_path = Path("datasets") / args.dataset
            if local_path.exists():
                print_info(f"Loading from: {Colors.BLUE}datasets/{args.dataset}{Colors.RESET}")
                dataset = load_dataset(args.dataset, root=local_path)
            else:
                print_info(f"Loading from: {Colors.BLUE}HuggingFace Hub{Colors.RESET}")
                dataset = load_dataset(args.dataset)
        info = dataset.info()

        print(f"\n{Colors.BOLD}Dataset Statistics:{Colors.RESET}")
        print(f"  Episodes:   {Colors.GREEN}{info['total_episodes']:>6}{Colors.RESET}")
        print(f"  Frames:     {Colors.GREEN}{info['total_frames']:>6}{Colors.RESET}")
        print(f"  FPS:        {info['fps']:>6}")
        print(f"  Robot:      {info['robot_type']}")

        print(f"\n{Colors.BOLD}Features ({len(info['features'])}):{Colors.RESET}")

        # Group features by type
        video_features = [f for f in info['features'] if f in info.get('video_keys', [])]
        other_features = [f for f in info['features'] if f not in info.get('video_keys', [])]

        if video_features:
            print(f"  {Colors.YELLOW}Video:{Colors.RESET}")
            for feature in video_features:
                shape = info['shapes'][feature]
                print(f"    • {feature:<28} {str(shape):<15} {Colors.YELLOW}video{Colors.RESET}")

        if other_features:
            print(f"  {Colors.GREEN}Data:{Colors.RESET}")
            for feature in other_features:
                shape = info['shapes'][feature]
                dtype = info['dtypes'][feature]
                print(f"    • {feature:<28} {str(shape):<15} {dtype}")

        print()
        print_success("Dataset loaded successfully")

    except Exception as e:
        print_error(f"Failed to load dataset: {e}")
        sys.exit(1)


def cmd_split(args):
    print_header(f"Splitting: {args.dataset}")

    if len(args.ratio) != len(args.names):
        print_error("Number of ratios must match number of names")
        sys.exit(1)

    if abs(sum(args.ratio) - 1.0) > 0.01:
        print_error(f"Ratios must sum to 1.0 (got {sum(args.ratio)})")
        sys.exit(1)

    splits_dict = {name: ratio for name, ratio in zip(args.names, args.ratio)}

    try:
        local_path = Path("datasets") / args.dataset
        root = local_path if local_path.exists() else None

        # Load original dataset to show info
        original = load_dataset(args.dataset, root=root)
        original_info = original.info()

        print(f"\n{Colors.BOLD}Original Dataset:{Colors.RESET}")
        print(f"  Episodes: {Colors.BLUE}{original_info['total_episodes']}{Colors.RESET}")
        print(f"  Frames:   {Colors.BLUE}{original_info['total_frames']}{Colors.RESET}")

        print(f"\n{Colors.BOLD}Split Plan:{Colors.RESET}")
        for name, ratio in splits_dict.items():
            expected_episodes = int(original_info['total_episodes'] * ratio)
            print(f"  {name:<10} {ratio:>5.1%}  →  ~{expected_episodes} episodes")

        print(f"\n{Colors.BLUE}ℹ{Colors.RESET} Output: datasets/{args.output}/")
        print()

        output_dir = Path("datasets") / args.output

        pbar = progress_bar("Splitting dataset")
        with suppress_output():
            splits = split_dataset_by_ratio(
                args.dataset,
                splits=splits_dict,
                output_dir=output_dir,
                root=root
            )
        if pbar:
            pbar.close()

        print(f"\n{Colors.BOLD}Split Results:{Colors.RESET}")
        for name, dataset in splits.items():
            info = dataset.info()
            episodes = info['total_episodes']
            frames = info['total_frames']
            ratio = splits_dict[name]
            print(f"  {Colors.GREEN}✓{Colors.RESET} {name:<10} {ratio:>5.1%}  →  {episodes:>4} episodes  {frames:>6} frames")

        print()
        print_success(f"Split complete: {len(splits)} datasets created")

    except Exception as e:
        print_error(f"Split failed: {e}")
        sys.exit(1)


def cmd_delete(args):
    print_header(f"Deleting episodes: {args.dataset}")

    episodes = [int(x.strip()) for x in args.episodes.split(',')]

    try:
        local_path = Path("datasets") / args.dataset
        root = local_path if local_path.exists() else None

        # Load original to show before/after
        original = load_dataset(args.dataset, root=root)
        original_info = original.info()

        print(f"\n{Colors.BOLD}Original Dataset:{Colors.RESET}")
        print(f"  Episodes: {Colors.BLUE}{original_info['total_episodes']}{Colors.RESET}")
        print(f"  Frames:   {Colors.BLUE}{original_info['total_frames']}{Colors.RESET}")

        print(f"\n{Colors.BOLD}Operation:{Colors.RESET}")
        print(f"  {Colors.RED}✗{Colors.RESET} Removing {len(episodes)} episodes: {Colors.RED}{episodes}{Colors.RESET}")
        remaining_count = original_info['total_episodes'] - len(episodes)
        print(f"  {Colors.GREEN}✓{Colors.RESET} Keeping {remaining_count} episodes")

        print(f"\n{Colors.BLUE}ℹ{Colors.RESET} Output: datasets/{args.output}/")
        print()

        output_dir = Path("datasets") / args.output

        pbar = progress_bar("Deleting episodes", total=len(episodes))
        with suppress_output():
            filtered = delete_episodes_from_dataset(
                args.dataset,
                episode_indices=episodes,
                new_repo_id=args.output,
                output_dir=output_dir,
                root=root
            )
        if pbar:
            pbar.update(len(episodes))
            pbar.close()

        filtered_info = filtered.info()
        print(f"\n{Colors.BOLD}Result:{Colors.RESET}")
        print(f"  Episodes: {Colors.GREEN}{filtered_info['total_episodes']}{Colors.RESET}")
        print(f"  Frames:   {Colors.GREEN}{filtered_info['total_frames']}{Colors.RESET}")

        print()
        print_success(f"Deleted {len(episodes)} episodes successfully")

    except Exception as e:
        print_error(f"Delete failed: {e}")
        sys.exit(1)


def cmd_add_feature(args):
    print_header(f"Adding feature: {args.dataset}")

    try:
        local_path = Path("datasets") / args.dataset
        root = local_path if local_path.exists() else None

        dataset = load_dataset(args.dataset, root=root)
        original_info = dataset.info()
        total_frames = original_info["total_frames"]

        print(f"\n{Colors.BOLD}Original Dataset:{Colors.RESET}")
        print(f"  Features: {Colors.BLUE}{len(original_info['features'])}{Colors.RESET}")
        print(f"  Frames:   {Colors.BLUE}{total_frames}{Colors.RESET}")

        print(f"\n{Colors.BOLD}New Feature:{Colors.RESET}")
        print(f"  Name:  {Colors.GREEN}{args.name}{Colors.RESET}")
        print(f"  Type:  {args.type}")
        print(f"  Shape: (1,)")
        print(f"  Dtype: float32")

        if args.type == "reward":
            data = np.random.randn(total_frames).astype(np.float32)
            schema = {"dtype": "float32", "shape": (1,), "names": None}
        elif args.type == "success":
            data = np.random.randint(0, 2, total_frames).astype(np.float32)
            schema = {"dtype": "float32", "shape": (1,), "names": None}
        else:
            data = np.random.randn(total_frames).astype(np.float32)
            schema = {"dtype": "float32", "shape": (1,), "names": None}

        print(f"\n{Colors.BLUE}ℹ{Colors.RESET} Output: datasets/{args.output}/")
        print()

        output_dir = Path("datasets") / args.output

        pbar = progress_bar("Adding feature", total=total_frames)
        with suppress_output():
            new_dataset = add_features_to_dataset(
                args.dataset,
                features={args.name: (data, schema)},
                new_repo_id=args.output,
                output_dir=output_dir,
                root=root
            )
        if pbar:
            pbar.update(total_frames)
            pbar.close()

        new_info = new_dataset.info()
        print(f"\n{Colors.BOLD}Result:{Colors.RESET}")
        print(f"  Features: {Colors.BLUE}{len(original_info['features'])}{Colors.RESET} → {Colors.GREEN}{len(new_info['features'])} (+1){Colors.RESET}")

        print()
        print_success(f"Feature '{args.name}' added successfully")

    except Exception as e:
        print_error(f"Add feature failed: {e}")
        sys.exit(1)


def cmd_episode_info(args):
    print_header(f"Episode {args.episode}: {args.dataset}")

    try:
        if args.stream:
            print_info(f"Streaming from: {Colors.BLUE}HuggingFace Hub (no local download){Colors.RESET}")
            dataset = load_dataset(args.dataset, stream=True)
        else:
            local_path = Path("datasets") / args.dataset
            if local_path.exists():
                dataset = load_dataset(args.dataset, root=local_path)
            else:
                dataset = load_dataset(args.dataset)

        dataset_info = dataset.info()
        episode_info = dataset.get_episode_info(args.episode)

        print(f"\n{Colors.BOLD}Episode Details:{Colors.RESET}")
        print(f"  Index:        {Colors.GREEN}{episode_info['episode_index']}{Colors.RESET}")
        print(f"  Length:       {Colors.GREEN}{episode_info['length']}{Colors.RESET} frames")
        print(f"  Frame Range:  {episode_info['dataset_from_index']} → {episode_info['dataset_to_index']}")

        # Calculate percentage of dataset
        percentage = (episode_info['length'] / dataset_info['total_frames']) * 100
        print(f"  Size:         {percentage:.2f}% of dataset")

        if 'tasks' in episode_info:
            print(f"\n{Colors.BOLD}Tasks:{Colors.RESET}")
            print(f"  {episode_info['tasks']}")

        print()
        print_success(f"Episode {args.episode} of {dataset_info['total_episodes']} episodes")

    except Exception as e:
        print_error(f"Failed: {e}")
        sys.exit(1)


def cmd_visualize(args):
    print_header(f"Visualizing: {args.dataset}")
    
    try:
        if args.stream:
            print_error("Cannot visualize streaming datasets. Visualization requires local files.")
            print_info("Please download the dataset first or remove --stream flag")
            sys.exit(1)
        
        # First, try to use lerobot's built-in visualization script if available
        try:
            import importlib
            lerobot_visualize = importlib.import_module("lerobot.scripts.visualize_dataset")
            # If we can import it, use lerobot's script
            print_info("Using lerobot's built-in visualization script")
            
            # Determine if it's a local dataset or remote
            dataset_path = None
            is_local = args.mode == "local" or args.root is not None
            
            # If root is provided, use it
            if args.root:
                dataset_path = Path(args.root).resolve()
                if not dataset_path.exists():
                    print_error(f"Root path not found: {dataset_path}")
                    sys.exit(1)
            else:
                # Check if it's a local path (absolute or relative)
                potential_path = Path(args.dataset)
                if potential_path.exists() and potential_path.is_dir():
                    dataset_path = potential_path.resolve()
                    is_local = True
                else:
                    # Check if it exists in datasets folder
                    local_path = Path("datasets") / args.dataset
                    if local_path.exists():
                        dataset_path = local_path.resolve()
                        is_local = True
            
            # Build command arguments for lerobot's script
            cmd = [sys.executable, "-m", "lerobot.scripts.visualize_dataset"]
            cmd.extend(["--repo-id", args.dataset])
            
            if is_local and dataset_path:
                cmd.extend(["--root", str(dataset_path)])
                cmd.extend(["--mode", "local"])
            
            if args.episode_index is not None:
                cmd.extend(["--episode-index", str(args.episode_index)])
            
            print()
            try:
                subprocess.run(cmd, check=True)
                return
            except subprocess.CalledProcessError as e:
                print_error(f"lerobot visualization script failed: {e}")
                print_info("Falling back to Rerun CLI method...")
                print()
        except (ImportError, ModuleNotFoundError):
            # lerobot's script not available, use our Rerun CLI method
            pass
        
        # Fallback: Use Rerun CLI directly
        rerun_cmd = shutil.which("rerun")
        if not rerun_cmd:
            print_error("Rerun CLI not found. Please install it:")
            print_info("  pip install rerun-sdk")
            print_info("  Or visit: https://rerun.io/docs/getting-started/installing")
            sys.exit(1)
        
        dataset_path = None
        
        # If root is provided, use it
        if args.root:
            dataset_path = Path(args.root).resolve()
            if not dataset_path.exists():
                print_error(f"Root path not found: {dataset_path}")
                sys.exit(1)
            print_info(f"Using local dataset: {Colors.BLUE}{dataset_path}{Colors.RESET}")
        else:
            # Check if it's a local path (absolute or relative)
            potential_path = Path(args.dataset)
            if potential_path.exists() and potential_path.is_dir():
                dataset_path = potential_path.resolve()
                print_info(f"Using local dataset: {Colors.BLUE}{dataset_path}{Colors.RESET}")
            else:
                # Check if it exists in datasets folder
                local_path = Path("datasets") / args.dataset
                if local_path.exists():
                    dataset_path = local_path.resolve()
                    print_info(f"Using local dataset: {Colors.BLUE}{dataset_path}{Colors.RESET}")
                else:
                    # Need to download from HuggingFace
                    print_info(f"Dataset not found locally. Downloading from HuggingFace Hub...")
                    print_info(f"  Repository: {Colors.BLUE}{args.dataset}{Colors.RESET}")
                    
                    # Try to download to datasets folder
                    repo_name = args.dataset.split("/")[-1]
                    output_dir = Path("datasets") / repo_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    print_info(f"  Downloading to: {Colors.BLUE}{output_dir}{Colors.RESET}")
                    print()
                    
                    pbar = progress_bar("Downloading dataset")
                    with suppress_output():
                        dataset = load_dataset(args.dataset, root=output_dir)
                        # Get the actual root path from the dataset
                        if hasattr(dataset.dataset, 'root') and dataset.dataset.root:
                            dataset_path = Path(dataset.dataset.root).resolve()
                        else:
                            dataset_path = output_dir.resolve()
                    
                    if pbar:
                        pbar.close()
                    
                    if not dataset_path.exists():
                        print_error(f"Could not locate downloaded dataset at {dataset_path}")
                        print_info("Please specify the local path directly or check the download location")
                        sys.exit(1)
                    
                    print_success(f"Dataset downloaded to: {dataset_path}")
        
        if not dataset_path or not dataset_path.exists():
            print_error(f"Dataset path not found: {dataset_path}")
            sys.exit(1)
        
        # Verify it's a valid LeRobot dataset (should have meta/ directory)
        meta_dir = dataset_path / "meta"
        if not meta_dir.exists():
            print_error(f"Invalid LeRobot dataset: missing 'meta' directory at {dataset_path}")
            print_info("Rerun requires a valid LeRobot dataset structure")
            sys.exit(1)
        
        print()
        print_success(f"Opening Rerun viewer for: {dataset_path}")
        if args.episode_index is not None:
            print_info(f"Episode: {args.episode_index}")
        print_info("Close the viewer window or press Ctrl+C to exit")
        print()
        
        # Call rerun CLI
        try:
            subprocess.run([rerun_cmd, str(dataset_path)], check=True)
        except KeyboardInterrupt:
            print()
            print_info("Visualization closed by user")
        except subprocess.CalledProcessError as e:
            print_error(f"Rerun failed: {e}")
            print_info("Make sure rerun-sdk is installed: pip install rerun-sdk")
            sys.exit(1)
        
    except Exception as e:
        print_error(f"Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_validate(args):
    print_header(f"Validating: {args.dataset}")

    try:
        if args.stream:
            print_info(f"Loading from: {Colors.BLUE}HuggingFace Hub (streaming){Colors.RESET}")
            dataset = load_dataset(args.dataset, stream=True)
        else:
            local_path = Path("datasets") / args.dataset
            if local_path.exists():
                print_info(f"Loading from: {Colors.BLUE}datasets/{args.dataset}{Colors.RESET}")
                dataset = load_dataset(args.dataset, root=local_path, download_videos=False)
            else:
                print_info(f"Loading from: {Colors.BLUE}HuggingFace Hub{Colors.RESET}")
                dataset = load_dataset(args.dataset, download_videos=False)

        info = dataset.info()
        print(f"\n{Colors.BOLD}Dataset Info:{Colors.RESET}")
        print(f"  Episodes: {Colors.BLUE}{info['total_episodes']}{Colors.RESET}")
        print(f"  Frames:   {Colors.BLUE}{info['total_frames']}{Colors.RESET}")
        print()

        print_info("Running validation checks...")
        print()

        pbar = progress_bar("Validating dataset")

        report = dataset.validate(
            check_semantic=not args.skip_semantic,
            check_action_normalization=not args.skip_action_norm,
            check_observation_consistency=not args.skip_obs_consistency,
            check_frame_alignment=not args.skip_frame_alignment,
            check_off_by_one=not args.skip_off_by_one,
        )

        if pbar:
            pbar.close()

        # Print summary
        print(f"\n{Colors.BOLD}Validation Results:{Colors.RESET}")
        print("=" * 60)

        total_checks = len(report.results)
        error_count = len(report.errors)
        warning_count = len(report.warnings)
        passed_count = sum(1 for r in report.results if r.passed)

        print(f"Total Checks:  {total_checks}")
        print(f"Passed:        {Colors.GREEN}{passed_count}{Colors.RESET}")
        print(f"Errors:        {Colors.RED if error_count > 0 else Colors.GREEN}{error_count}{Colors.RESET}")
        print(f"Warnings:      {Colors.YELLOW if warning_count > 0 else Colors.GREEN}{warning_count}{Colors.RESET}")
        print()

        # Print errors
        if report.errors:
            print(f"{Colors.BOLD}{Colors.RED}ERRORS:{Colors.RESET}")
            for result in report.errors:
                print(f"  {Colors.RED}✗{Colors.RESET} {result.message}")
                if args.verbose and result.details:
                    for key, value in result.details.items():
                        if isinstance(value, list) and len(value) > 3:
                            print(f"    {key}: {value[:3]}... (showing first 3)")
                        else:
                            print(f"    {key}: {value}")
            print()

        # Print warnings
        if report.warnings:
            print(f"{Colors.BOLD}{Colors.YELLOW}WARNINGS:{Colors.RESET}")
            for result in report.warnings:
                print(f"  {Colors.YELLOW}⚠{Colors.RESET} {result.message}")
                if args.verbose and result.details:
                    for key, value in result.details.items():
                        if isinstance(value, list) and len(value) > 3:
                            print(f"    {key}: {value[:3]}... (showing first 3)")
                        else:
                            print(f"    {key}: {value}")
            print()

        # Print info messages
        if args.verbose and report.info:
            print(f"{Colors.BOLD}INFO:{Colors.RESET}")
            for result in report.info:
                print(f"  {Colors.BLUE}ℹ{Colors.RESET} {result.message}")
                if result.details:
                    for key, value in result.details.items():
                        if isinstance(value, list) and len(value) > 3:
                            print(f"    {key}: {value[:3]}... (showing first 3)")
                        else:
                            print(f"    {key}: {value}")
            print()

        # Overall result
        if report.passed:
            print_success("Validation passed! Dataset is ready to use.")
        else:
            if error_count > 0:
                print_error(f"Validation failed with {error_count} error(s)")
                sys.exit(1)
            else:
                print_success("Validation passed with warnings")

    except Exception as e:
        print_error(f"Validation failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


def cmd_merge(args):
    print_header("Merging Datasets")

    datasets_list = [d.strip() for d in args.datasets.split(',')]

    if len(datasets_list) < 2:
        print_error("Need at least 2 datasets to merge")
        sys.exit(1)

    print_info(f"Datasets to merge: {len(datasets_list)}")
    for i, ds in enumerate(datasets_list, 1):
        print(f"  {i}. {ds}")

    print_info("\nLoading datasets...")
    try:
        loaded_datasets = {}
        dataset_roots = []

        for ds in datasets_list:
            local_path = Path("datasets") / ds
            if local_path.exists():
                print_info(f"  Loading local: {ds}")
            else:
                print_info(f"  Loading remote: {ds}")
        print()

        load_pbar = progress_bar("Loading datasets", total=len(datasets_list))
        for ds in datasets_list:
            local_path = Path("datasets") / ds
            if local_path.exists():
                loaded_datasets[ds] = load_dataset(ds, root=local_path)
                dataset_roots.append(local_path)
            else:
                loaded_datasets[ds] = load_dataset(ds)
                dataset_roots.append(None)
            if load_pbar:
                load_pbar.update(1)
        if load_pbar:
            load_pbar.close()
        print()

        first_dataset = list(loaded_datasets.values())[0]
        first_features = set(first_dataset.info()['features'])
        first_shapes = first_dataset.info()['shapes']
        first_dtypes = first_dataset.info()['dtypes']

        all_compatible = True
        differences = []

        for ds_name, dataset in list(loaded_datasets.items())[1:]:
            ds_features = set(dataset.info()['features'])
            ds_shapes = dataset.info()['shapes']
            ds_dtypes = dataset.info()['dtypes']

            missing = first_features - ds_features
            extra = ds_features - first_features

            if missing or extra:
                all_compatible = False
                if missing:
                    differences.append(f"  {Colors.RED}✗{Colors.RESET} {ds_name} missing: {', '.join(missing)}")
                if extra:
                    differences.append(f"  {Colors.YELLOW}!{Colors.RESET} {ds_name} has extra: {', '.join(extra)}")

            common_features = first_features & ds_features
            for feature in common_features:
                if first_shapes[feature] != ds_shapes[feature]:
                    all_compatible = False
                    differences.append(
                        f"  {Colors.RED}✗{Colors.RESET} {ds_name}.{feature} shape mismatch: "
                        f"{first_shapes[feature]} vs {ds_shapes[feature]}"
                    )
                if first_dtypes[feature] != ds_dtypes[feature]:
                    all_compatible = False
                    differences.append(
                        f"  {Colors.RED}✗{Colors.RESET} {ds_name}.{feature} dtype mismatch: "
                        f"{first_dtypes[feature]} vs {ds_dtypes[feature]}"
                    )

        if not all_compatible:
            print_error("Datasets have incompatible features:")
            for diff in differences:
                print(diff)
            print()
            print_error("Cannot merge: Features must be identical")
            print_info("Tip: Use 'add-feature' or 'remove-feature' to align datasets first")
            sys.exit(1)

        print_success("All datasets compatible!")

        # Compare metadata
        all_fps = [ds.info()['fps'] for ds in loaded_datasets.values()]
        all_robots = [ds.info()['robot_type'] for ds in loaded_datasets.values()]
        fps_match = len(set(all_fps)) == 1
        robot_match = len(set(all_robots)) == 1

        print(f"\n{Colors.BOLD}Dataset Summary:{Colors.RESET}")
        print(f"  Features:        {Colors.GREEN}{len(first_features)} ✓{Colors.RESET} (all compatible)")

        total_episodes = sum(ds.info()['total_episodes'] for ds in loaded_datasets.values())
        total_frames = sum(ds.info()['total_frames'] for ds in loaded_datasets.values())
        print(f"  Total Episodes:  {total_episodes}")
        print(f"  Total Frames:    {total_frames}")

        # Color-coded metadata comparison
        fps_color = Colors.GREEN if fps_match else Colors.YELLOW
        fps_icon = "✓" if fps_match else "~"
        robot_color = Colors.GREEN if robot_match else Colors.YELLOW
        robot_icon = "✓" if robot_match else "~"

        print(f"  FPS:             {fps_color}{all_fps[0]} {fps_icon}{Colors.RESET}", end="")
        if not fps_match:
            print(f" {Colors.YELLOW}(varies: {', '.join(map(str, set(all_fps)))}){Colors.RESET}")
        else:
            print()

        print(f"  Robot Type:      {robot_color}{all_robots[0]} {robot_icon}{Colors.RESET}", end="")
        if not robot_match:
            print(f" {Colors.YELLOW}(varies: {', '.join(set(all_robots))}){Colors.RESET}")
        else:
            print()

        print(f"\n{Colors.BOLD}Datasets to merge:{Colors.RESET}")
        for ds_name, dataset in loaded_datasets.items():
            info = dataset.info()
            print(f"  {Colors.BLUE}•{Colors.RESET} {ds_name:<30} {info['total_episodes']:>4} episodes  {info['total_frames']:>6} frames")

        if not args.force:
            print()
            response = input(f"{Colors.YELLOW}⚠{Colors.RESET}  Continue with merge? [y/N]: ")
            if response.lower() != 'y':
                print_info("Merge cancelled")
                sys.exit(0)

        print()

        output_dir = Path("datasets") / args.output

        merge_pbar = progress_bar("Merging datasets", total=total_episodes)
        with suppress_output():
            merged = merge_datasets_wrapper(
                repo_ids=datasets_list,
                output_repo_id=args.output,
                output_dir=output_dir,
                roots=dataset_roots
            )
        if merge_pbar:
            merge_pbar.update(total_episodes)
            merge_pbar.close()

        merged_info = merged.info()

        print(f"\n{Colors.BOLD}Merge Complete!{Colors.RESET}")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Combined {len(datasets_list)} datasets")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Result: {merged_info['total_episodes']} episodes, {merged_info['total_frames']} frames")
        print(f"  {Colors.GREEN}✓{Colors.RESET} Saved to: datasets/{args.output}/")

        print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
        print(f"  {'Datasets merged:':<20} {len(datasets_list)}")
        print(f"  {'Total episodes:':<20} {Colors.BLUE}{total_episodes}{Colors.RESET} → {Colors.GREEN}{merged_info['total_episodes']}{Colors.RESET}")
        print(f"  {'Total frames:':<20} {Colors.BLUE}{total_frames}{Colors.RESET} → {Colors.GREEN}{merged_info['total_frames']}{Colors.RESET}")
        print()

    except Exception as e:
        print_error(f"Merge failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='LeRobot Dataset Tool - CLI for dataset manipulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    info_parser = subparsers.add_parser('info', help='Get dataset information')
    info_parser.add_argument('--dataset', required=True, help='Dataset repo ID or local path')
    info_parser.add_argument('--stream', action='store_true', help='Stream dataset directly from Hub without local copies')

    split_parser = subparsers.add_parser('split', help='Split dataset')
    split_parser.add_argument('--dataset', required=True, help='Dataset repo ID or local path')
    split_parser.add_argument('--ratio', nargs='+', type=float, required=True, help='Split ratios')
    split_parser.add_argument('--names', nargs='+', required=True, help='Split names')
    split_parser.add_argument('--output', default='split', help='Output directory name')

    delete_parser = subparsers.add_parser('delete', help='Delete episodes')
    delete_parser.add_argument('--dataset', required=True, help='Dataset repo ID or local path')
    delete_parser.add_argument('--episodes', required=True, help='Comma-separated episode indices')
    delete_parser.add_argument('--output', default='filtered', help='Output directory name')

    feature_parser = subparsers.add_parser('add-feature', help='Add feature to dataset')
    feature_parser.add_argument('--dataset', required=True, help='Dataset repo ID or local path')
    feature_parser.add_argument('--name', required=True, help='Feature name')
    feature_parser.add_argument('--type', choices=['reward', 'success', 'custom'], default='custom', help='Feature type')
    feature_parser.add_argument('--output', default='with_feature', help='Output directory name')

    episode_parser = subparsers.add_parser('episode', help='Get episode information')
    episode_parser.add_argument('--dataset', required=True, help='Dataset repo ID or local path')
    episode_parser.add_argument('--episode', type=int, required=True, help='Episode index')
    episode_parser.add_argument('--stream', action='store_true', help='Stream dataset directly from Hub without local copies')

    merge_parser = subparsers.add_parser('merge', help='Merge multiple datasets')
    merge_parser.add_argument('--datasets', required=True, help='Comma-separated dataset repo IDs or local paths')
    merge_parser.add_argument('--output', default='merged', help='Output directory name')
    merge_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')

    visualize_parser = subparsers.add_parser('visualize', help='Visualize dataset with Rerun')
    visualize_parser.add_argument('--dataset', '--repo-id', dest='dataset', required=True, help='Dataset repo ID or local path')
    visualize_parser.add_argument('--root', type=str, default=None, help='Root directory for local datasets')
    visualize_parser.add_argument('--episode-index', type=int, default=None, help='Episode index to visualize (0-based)')
    visualize_parser.add_argument('--mode', type=str, default=None, help='Mode: "local" for local datasets')
    visualize_parser.add_argument('--stream', action='store_true', help='Stream dataset (note: visualization requires local files)')

    validate_parser = subparsers.add_parser('validate', help='Validate dataset for common issues')
    validate_parser.add_argument('--dataset', required=True, help='Dataset repo ID or local path')
    validate_parser.add_argument('--stream', action='store_true', help='Stream dataset directly from Hub without local copies')
    validate_parser.add_argument('--skip-semantic', action='store_true', help='Skip semantic validation')
    validate_parser.add_argument('--skip-action-norm', action='store_true', help='Skip action normalization checks')
    validate_parser.add_argument('--skip-obs-consistency', action='store_true', help='Skip observation consistency checks')
    validate_parser.add_argument('--skip-frame-alignment', action='store_true', help='Skip frame alignment checks')
    validate_parser.add_argument('--skip-off-by-one', action='store_true', help='Skip off-by-one error detection')
    validate_parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed validation results')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        'info': cmd_info,
        'split': cmd_split,
        'delete': cmd_delete,
        'add-feature': cmd_add_feature,
        'episode': cmd_episode_info,
        'merge': cmd_merge,
        'visualize': cmd_visualize,
        'validate': cmd_validate,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
