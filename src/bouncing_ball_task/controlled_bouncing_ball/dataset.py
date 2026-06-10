"""Dataset generation orchestrator for controlled bouncing ball experiments.

This module provides the main interface for generating controlled datasets
using a change vector manipulation approach for precise experimental control.
"""
import copy
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from bouncing_ball_task import index
from bouncing_ball_task.bouncing_ball import BouncingBallTask
from bouncing_ball_task.human_bouncing_ball import dataset as hds
from bouncing_ball_task.utils import logutils, pyutils, htaskutils
from bouncing_ball_task.controlled_bouncing_ball import defaults
from bouncing_ball_task.controlled_bouncing_ball.no_change import generate_no_change_trials


# Dictionary of available trial generation functions
# Import each trial type generator and add to this dict
dict_trial_type_generation_funcs = {
    "no_change": generate_no_change_trials,
}


def generate_controlled_dataset(
    controlled_dataset_parameters,
    task_parameters,
    shuffle=True,
    dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
    defaults_module=defaults,
):
    """Generate controlled bouncing ball dataset using change vector manipulation.
    
    This function generates a controlled dataset by:
    1. Creating base sequences with general parameters
    2. Extracting positions and change vectors
    3. Allowing variants to modify change vectors
    4. Generating color sequences from modified change vectors
    5. Creating final dataset with multiple color rotations
    
    Args:
        controlled_dataset_parameters: Dictionary of dataset generation parameters
        task_parameters: Dictionary of task-specific parameters
        shuffle: Whether to shuffle the generated trials
        dict_trial_type_generation_funcs: Dictionary mapping trial types to generation functions
        defaults_module: Module containing default parameters

    Returns:
        tuple: (task, output_samples, output_model_samples, output_targets, df_data, dict_metadata)
    """
    # Extract parameters
    num_base_sequences = controlled_dataset_parameters.get('num_base_sequences', 100)
    total_videos = controlled_dataset_parameters.get('total_videos', None)
    duration = controlled_dataset_parameters.get('duration', defaults_module.duration)
    variable_length = controlled_dataset_parameters.get('variable_length', False)
    
    # If total_videos specified, calculate num_base_sequences
    num_variants = len(dict_trial_type_generation_funcs)
    num_colors = 3  # RGB initial colors
    
    if total_videos is not None:
        num_base_sequences = total_videos // (num_variants * num_colors)
        logger.info(f"Calculated {num_base_sequences} base sequences for {total_videos} total videos")
    
    # Step 1: Generate N base sequences using BouncingBallTask
    base_task_parameters = copy.deepcopy(task_parameters)
    base_task_parameters['batch_size'] = num_base_sequences
    base_task_parameters['return_change'] = True
    base_task_parameters['return_change_mode'] = 'source'  # Get full change arrays
    
    # Thread the dataset seed INTO the base task so it seeds deterministically
    # (P0-2). Passing seed=None lets BouncingBallTask draw a fresh seed and report
    # it via resolved_seed; passing an int makes generation deterministic.
    seed = controlled_dataset_parameters.get('seed')
    base_task_parameters['seed'] = seed

    # Create task and generate sequences
    base_task = BouncingBallTask(**base_task_parameters)

    # Capture the seed the task actually resolved, for reproducible provenance.
    resolved_seed = base_task.resolved_seed
    
    # Get the base targets which include positions and change vectors
    base_targets = base_task.targets  # Shape: (N, T, 9)
    base_samples = base_task.samples  # Shape: (N, T, 5)
    
    # Step 2: Extract positions and change vectors
    positions = base_targets[:, :, :2]  # Shape: (N, T, 2)
    change_vectors = base_targets[:, :, 5:]  # Shape: (N, T, 4)
    
    # Step 3: Zero out color change features for controlled experiments
    change_vectors[:, :, -2:] = 0  # Zero out contingent and random color changes
    
    # Step 4: Process each variant
    all_change_vectors = []
    all_metadata = []
    variant_names = []
    
    for variant_name, variant_func in dict_trial_type_generation_funcs.items():
        # Each variant function modifies change vectors
        change_vectors_variant, dict_meta_variant = variant_func(
            change_vectors.copy(),
            controlled_dataset_parameters,
            task_parameters,
            base_task,
        )
        
        all_change_vectors.append(change_vectors_variant)
        all_metadata.append(dict_meta_variant)
        variant_names.extend([variant_name] * num_base_sequences)
    
    # Step 5: Concatenate variant change vectors
    change_array_variants = np.concatenate(all_change_vectors, axis=0)  # Shape: (N*V, T, 4)
    
    # Step 6: Generate color sequences from change vectors
    # Detect where any color change occurs
    color_change_sequences = np.any(change_array_variants[:, :, -2:], axis=-1)  # Shape: (N*V, T)
    
    # Create base color sequences as cumulative sum of changes
    color_sequences_base = np.cumsum(color_change_sequences, axis=1)  # Shape: (N*V, T)
    
    # Step 7: Create color rotations (start with each of 3 colors)
    all_color_sequences = []
    all_positions = []
    all_change_vectors_final = []
    all_variant_names = []
    all_base_indices = []
    all_color_indices = []
    
    for color_start in range(num_colors):
        # Apply color rotation
        color_sequences_categorical = (color_start + color_sequences_base) % num_colors
        
        # Convert to one-hot RGB representation
        color_sequences_rgb = np.eye(3)[color_sequences_categorical] * 255  # Shape: (N*V, T, 3)
        
        # Store data for this color rotation
        all_color_sequences.append(color_sequences_rgb)
        all_positions.append(np.tile(positions, (num_variants, 1, 1)))
        all_change_vectors_final.append(change_array_variants)
        all_variant_names.extend(variant_names)
        
        # Track indices
        base_indices = np.tile(np.arange(num_base_sequences), num_variants)
        all_base_indices.extend(base_indices)
        all_color_indices.extend([color_start] * len(base_indices))
    
    # Step 8: Combine all data
    final_positions = np.concatenate(all_positions, axis=0)  # Shape: (N*V*C, T, 2)
    final_colors = np.concatenate(all_color_sequences, axis=0)  # Shape: (N*V*C, T, 3)
    final_change_vectors = np.concatenate(all_change_vectors_final, axis=0)  # Shape: (N*V*C, T, 4)
    
    # Step 9: Create final targets by combining positions, colors, and change vectors
    final_targets = np.concatenate([
        final_positions,
        final_colors,
        final_change_vectors
    ], axis=-1)  # Shape: (N*V*C, T, 9)
    
    # Step 10: Generate samples by applying grayzone masks
    # Use vectorized version for efficiency
    final_samples = np.zeros((final_targets.shape[0], final_targets.shape[1], 5))
    
    # Extract positions and colors
    positions = final_targets[:, :, :2]
    colors = final_targets[:, :, 2:5]
    
    # Apply masking for each sequence
    for i in range(final_targets.shape[0]):
        # Create samples for this sequence using vectorized approach
        x_positions = positions[i, :, 0]
        ball_radius = base_task.ball_radius
        mask_start = base_task.mask_start
        mask_end = base_task.mask_end
        mask_color = np.array(base_task.mask_color)
        
        # Compute overlaps
        overlap_left = np.maximum(x_positions - ball_radius, mask_start)
        overlap_right = np.minimum(x_positions + ball_radius, mask_end)
        overlap_width = np.clip(overlap_right - overlap_left, 0, None)
        
        # Compute masked colors
        overlap_proportion = overlap_width / (2 * ball_radius)
        masked_colors = (1 - overlap_proportion[:, np.newaxis]) * colors[i] + \
                       overlap_proportion[:, np.newaxis] * mask_color
        
        # Combine position and masked color
        final_samples[i] = np.concatenate([positions[i], masked_colors], axis=1)
    
    # Step 11: Create preset task with generated data
    final_task_parameters = copy.deepcopy(task_parameters)
    final_task_parameters['batch_size'] = final_targets.shape[0]
    final_task_parameters['sequence_mode'] = 'preset'
    
    task = BouncingBallTask(
        **final_task_parameters,
        samples=final_samples,
        targets=final_targets
    )
    
    # Step 12: Generate metadata
    dict_metadata = generate_controlled_metadata(
        task,
        controlled_dataset_parameters,
        task_parameters,
        all_metadata,
        num_base_sequences,
        num_variants,
        num_colors,
        dict_trial_type_generation_funcs,
    )
    
    # Step 13: Create DataFrame with trial information
    df_data = generate_controlled_dataframe(
        task,
        final_targets,
        final_samples,
        all_variant_names,
        all_base_indices,
        all_color_indices,
        dict_metadata,
        duration,
        variable_length,
    )
    
    # Shuffle if requested
    if shuffle:
        indices = np.random.permutation(len(df_data))
        df_data = df_data.iloc[indices].reset_index(drop=True)
        final_samples = final_samples[indices]
        final_targets = final_targets[indices]
        
        # Update task with shuffled data
        task = BouncingBallTask(
            **final_task_parameters,
            samples=final_samples,
            targets=final_targets
        )
    
    # For compatibility with existing code, create model samples (same as samples for controlled)
    output_model_samples = final_samples.copy()
    
    return task, final_samples, output_model_samples, final_targets, df_data, dict_metadata


def generate_controlled_dataset_with_videos(
    controlled_dataset_parameters,
    task_parameters,
    output_dir,
    shuffle=True,
    dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
    defaults_module=defaults,
):
    """Generate controlled dataset and save videos - convenience function for tests.
    
    This function generates a controlled dataset and saves the videos,
    useful for tests that expect video files to be created.
    
    Args:
        controlled_dataset_parameters: Dictionary of dataset generation parameters
        task_parameters: Dictionary of task-specific parameters
        output_dir: Directory to save the dataset
        shuffle: Whether to shuffle the generated trials
        dict_trial_type_generation_funcs: Dictionary mapping trial types to generation functions
        defaults_module: Module containing default parameters
        
    Returns:
        tuple: (task, output_samples, output_model_samples, output_targets, df_data, dict_metadata)
    """
    # Generate the dataset
    task, output_samples, output_model_samples, output_targets, df_data, dict_metadata = generate_controlled_dataset(
        controlled_dataset_parameters,
        task_parameters,
        shuffle=shuffle,
        dict_trial_type_generation_funcs=dict_trial_type_generation_funcs,
        defaults_module=defaults_module,
    )
    
    # Save videos
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        dataset_name = dict_metadata["name"]
        
        path_videos = hds.save_video_dataset(
            output_dir,
            dataset_name,
            df_data,
            dict_metadata,
            output_samples,
            output_model_samples,
            output_targets,
            task,
            duration=controlled_dataset_parameters.get("duration", defaults_module.ControlledDatasetParameters().duration),
            mode=defaults_module.mode,
            multiplier=defaults_module.multiplier,
            save_target=True,
            save_animation=True,
            display_animation=defaults_module.display_animation,
            num_sequences=1,
            as_mp4=True,
            include_timestep=defaults_module.include_timestep,
            return_path=True,
            dryrun=False,
        )
        
        dict_metadata["dataset_dir"] = output_dir / dataset_name
    
    return task, output_samples, output_model_samples, output_targets, df_data, dict_metadata


def generate_controlled_metadata(
    task,
    controlled_dataset_parameters,
    task_parameters,
    variant_metadata,
    num_base_sequences,
    num_variants,
    num_colors,
    dict_trial_type_generation_funcs,
):
    """Generate metadata dictionary for controlled dataset.
    
    Args:
        task: BouncingBallTask instance
        controlled_dataset_parameters: Dataset parameters
        task_parameters: Task parameters
        variant_metadata: List of metadata from each variant
        num_base_sequences: Number of base sequences
        num_variants: Number of variants
        num_colors: Number of color rotations
        
    Returns:
        dict: Complete metadata dictionary
    """
    # Generate timestamp and seed
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    seed = controlled_dataset_parameters.get('seed', 0)  # Get seed from parameters
    
    # Generate name based on variants
    variant_str = "_".join(sorted(dict_trial_type_generation_funcs.keys()))
    if len(dict_trial_type_generation_funcs) == 1 and "no_change" in dict_trial_type_generation_funcs:
        # Special case for no_change only to maintain compatibility
        name = f"controlled_no_change_{timestamp}_{seed}"
    else:
        name = f"controlled_{variant_str}_{timestamp}_{seed}"
    
    dict_metadata = {
        # Basic info
        "name": name,
        "seed": seed,
        "dataset_type": "controlled",
        "timestamp": timestamp,
        "total_trials": num_base_sequences * num_variants * num_colors,
        "video_length_max_f": task.sequence_length,
        
        # Dataset generation parameters
        "controlled_dataset_parameters": controlled_dataset_parameters,
        "task_parameters": task_parameters,
        
        # Controlled-specific metadata
        "controlled_parameters": {
            "generation_method": "change_vector_manipulation",
            "num_base_sequences": num_base_sequences,
            "num_variants": num_variants,
            "num_colors": num_colors,
            "variants": list(dict_trial_type_generation_funcs.keys()),
            "color_changes_allowed": False,
            "base_sequence_reuse": num_variants * num_colors,
        },
        
        # Generation process info
        "base_generation_info": {
            "method": "Base sequences with change vector manipulation",
            "change_vector_shape": (num_base_sequences, task.sequence_length, 4),
            "color_sequence_generation": "Cumulative sum of change indicators with rotation",
            "mask_application": "Standard grayzone masking",
        },
    }
    
    # Add variant-specific metadata
    for i, (variant_name, meta) in enumerate(zip(dict_trial_type_generation_funcs.keys(), variant_metadata)):
        dict_metadata[variant_name] = {
            "num_trials": num_base_sequences * num_colors,
            "variant_metadata": meta,
            "task_parameters": task_parameters.copy(),
        }
    
    return dict_metadata


def generate_controlled_dataframe(
    task,
    targets,
    samples,
    variant_names,
    base_indices,
    color_indices,
    dict_metadata,
    duration,
    variable_length,
):
    """Generate DataFrame with trial information for controlled dataset.
    
    Args:
        task: BouncingBallTask instance
        targets: Target array
        samples: Sample array
        variant_names: List of variant names for each trial
        base_indices: Indices of base sequences
        color_indices: Indices of color rotations
        dict_metadata: Metadata dictionary
        duration: Video duration in ms
        variable_length: Whether to use variable length videos
        
    Returns:
        pd.DataFrame: Trial information
    """
    num_trials = targets.shape[0]
    
    # Calculate video lengths
    if variable_length:
        # Use exponential distribution for variable lengths
        exp_scale = duration / 1000  # Convert ms to seconds
        lengths = np.random.exponential(exp_scale, num_trials)
        lengths = np.clip(lengths, 1, task.sequence_length / 30)  # Clip to valid range
        lengths_frames = (lengths * 30).astype(int)  # Convert to frames
    else:
        lengths_frames = np.full(num_trials, int(duration * 30 / 1000))  # Fixed length
    
    # Extract final colors from targets
    final_colors_rgb = targets[:, -1, 2:5]
    final_color_names = []
    for rgb in final_colors_rgb:
        if np.allclose(rgb, [255, 0, 0]):
            final_color_names.append("red")
        elif np.allclose(rgb, [0, 255, 0]):
            final_color_names.append("green")
        elif np.allclose(rgb, [0, 0, 255]):
            final_color_names.append("blue")
        else:
            raise ValueError(f"Unrecognized RGB value: {rgb}")
    
    # Create base dataframe
    df_data = pd.DataFrame({
        # Core identification
        "idx": np.arange(num_trials),
        "length": lengths_frames,
        "trial": variant_names,
        "variant": pd.Categorical(variant_names),
        "base_sequence_idx": base_indices,
        "color_rotation_idx": color_indices,
        
        # Position/velocity indices (placeholder values for now)
        "idx_x_position": 0,
        "idx_y_positions": 0,
        "idx_velocity_y": 0,
        "idx_time": 0,
        
        # Color information
        "Final Color": final_color_names,
        
        # Fixed probability parameters
        "PCCNVC": 0.0,
        "PCCOVC": 0.0,
        "PVC": dict_metadata.get('task_parameters', {}).get('probability_velocity_change', 0.0),
        "PCCNVC_effective": 0.0,
        "PCCOVC_effective": 0.0,
        "PVC_effective": dict_metadata.get('task_parameters', {}).get('probability_velocity_change', 0.0),
        
        # Compatibility columns
        "Hazard Rate": pd.Categorical(["No Change"] * num_trials, categories=["No Change"]),
        "Contingency": pd.Categorical(["None"] * num_trials, categories=["None"]),
    })
    
    # Calculate change statistics from actual data
    bounces = np.zeros(num_trials)
    random_bounces = np.zeros(num_trials)
    color_change_bounce = np.zeros(num_trials)
    color_change_random = np.zeros(num_trials)
    
    for i in range(num_trials):
        # Count changes in the change vectors
        change_vec = targets[i, :lengths_frames[i], 5:]
        bounces[i] = np.sum(change_vec[:, 0])  # Bounce velocity changes
        random_bounces[i] = np.sum(change_vec[:, 1])  # Random velocity changes
        color_change_bounce[i] = np.sum(change_vec[:, 2])  # Color changes on bounce
        color_change_random[i] = np.sum(change_vec[:, 3])  # Random color changes
    
    # Add change statistics
    df_data["Bounces"] = bounces.astype(int)
    df_data["Random Bounces"] = random_bounces.astype(int)
    df_data["Color Change Bounce"] = color_change_bounce.astype(int)
    df_data["Color Change Random"] = color_change_random.astype(int)
    df_data["Total Changes"] = (bounces + random_bounces + color_change_bounce + color_change_random).astype(int)
    
    return df_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Inferred args from the dictionaries
    parser = pyutils.add_dataclass_args(parser, defaults.ControlledTaskParameters)
    parser = pyutils.add_dataclass_args(parser, defaults.ControlledDatasetParameters)

    # Manual additions
    parser.add_argument("--dir_base", type=Path, default=index.dir_repo/"data/raw/bb_datasets/controlled")
    parser.add_argument("--name_dataset", default=defaults.name_dataset)
    parser.add_argument("--display_animation", default=defaults.display_animation)
    parser.add_argument("--mode", type=str, default=defaults.mode)
    parser.add_argument("--multiplier", type=int, default=defaults.multiplier)
    parser.add_argument("--include_timestep", default=defaults.include_timestep)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    
    # Parse the arguments from the command line
    args = parser.parse_args()
    # Setup the logger
    logger = logutils.configure_logger(verbose=args.verbose, trace=args.debug)
    dir_base = Path(args.dir_base)
    
    task_parameters = {
        key: getattr(args, key) for key in defaults.ControlledTaskParameters.keys
    }
    controlled_dataset_parameters = {
        key: getattr(args, key) for key in defaults.ControlledDatasetParameters.keys
    }

    # Generate the dataset
    task, output_samples, output_model_samples, output_targets, df_data, dict_metadata = generate_controlled_dataset(
        controlled_dataset_parameters,
        task_parameters,
        shuffle=False,
    )

    # Generate dataset name
    dict_metadata["name"] = name_dataset = htaskutils.generate_dataset_name(
        args.name_dataset,
        seed=dict_metadata["seed"],
    )    

    # Save the dataset
    path_videos = hds.save_video_dataset(
        dir_base,
        name_dataset,
        df_data,
        dict_metadata,
        output_samples,
        output_model_samples,
        output_targets,
        task,
        duration=args.duration,
        mode=args.mode,
        multiplier=args.multiplier,
        save_target=True,
        save_animation=True,
        display_animation=args.display_animation,
        num_sequences=1,
        as_mp4=True,
        include_timestep=args.include_timestep,
        return_path=True,
        dryrun=args.dryrun,
    )
    
    logger.info(f"Controlled dataset saved to: {dir_base / name_dataset}")
