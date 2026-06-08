"""End-product tests for controlled dataset generation.

These tests validate the final output of the controlled dataset generation,
focusing on critical properties like position consistency and color uniqueness.
"""
import pytest
import numpy as np
import pandas as pd
from collections import defaultdict

from bouncing_ball_task.controlled_bouncing_ball.dataset import generate_controlled_dataset
from bouncing_ball_task.controlled_bouncing_ball.no_change import generate_no_change_trials


@pytest.mark.controlled_dataset
class TestControlledDatasetEndProduct:
    """Test the end product of controlled dataset generation."""
    
    def test_position_consistency_across_variants(self, controlled_dataset_small):
        """CRITICAL: Verify all variants from same base have identical positions at all timesteps."""
        targets = controlled_dataset_small['targets']
        df_data = controlled_dataset_small['df_data']
        
        # Group trials by base_sequence_idx
        base_groups = defaultdict(list)
        for idx, row in df_data.iterrows():
            base_idx = row['base_sequence_idx']
            base_groups[base_idx].append(idx)
        
        # For each base sequence group
        for base_idx, trial_indices in base_groups.items():
            if len(trial_indices) < 2:
                continue  # Need at least 2 trials to compare
            
            # Get the first trial's positions as reference
            reference_positions = targets[trial_indices[0], :, :2]  # (T, 2)
            
            # Compare all other trials from same base
            for trial_idx in trial_indices[1:]:
                trial_positions = targets[trial_idx, :, :2]
                
                # Positions should be exactly identical
                np.testing.assert_array_equal(
                    reference_positions,
                    trial_positions,
                    err_msg=f"Positions differ for trials from base sequence {base_idx}: "
                            f"trial {trial_indices[0]} vs trial {trial_idx}"
                )
    
    def test_color_uniqueness_per_variant(self, controlled_dataset_small):
        """CRITICAL: Verify each variant produces all 3 unique final colors."""
        df_data = controlled_dataset_small['df_data']
        
        # Group by variant
        variant_groups = df_data.groupby('variant')
        
        for variant_name, variant_df in variant_groups:
            # Get unique final colors for this variant
            unique_colors = set(variant_df['Final Color'].unique())
            expected_colors = {'red', 'green', 'blue'}
            
            # Each variant should have all 3 colors
            assert unique_colors == expected_colors, \
                f"Variant '{variant_name}' missing colors: expected {expected_colors}, got {unique_colors}"
            
            # Check that we have the right count of each color
            color_counts = variant_df['Final Color'].value_counts()
            
            # For a balanced dataset, each color should appear roughly equally
            # With num_base_sequences=5, we expect 5 trials per color per variant
            expected_count = len(variant_df) // 3
            for color in expected_colors:
                actual_count = color_counts.get(color, 0)
                # Allow some variation due to rounding
                assert abs(actual_count - expected_count) <= 1, \
                    f"Variant '{variant_name}' has imbalanced color distribution: " \
                    f"{color} appears {actual_count} times, expected ~{expected_count}"
    
    def test_trial_count_formula(self, controlled_dataset_small):
        """Verify total trials = N_base × N_variants × N_colors."""
        df_data = controlled_dataset_small['df_data']
        metadata = controlled_dataset_small['metadata']
        controlled_params = controlled_dataset_small['controlled_params']
        
        # Extract counts
        num_base = controlled_params['num_base_sequences']
        num_variants = len(metadata['controlled_parameters']['variants'])
        num_colors = metadata['controlled_parameters']['num_colors']
        
        # Calculate expected total
        expected_total = num_base * num_variants * num_colors
        actual_total = len(df_data)
        
        assert actual_total == expected_total, \
            f"Trial count mismatch: expected {num_base}×{num_variants}×{num_colors}={expected_total}, " \
            f"got {actual_total}"
    
    def test_color_rotation_indices(self, controlled_dataset_small):
        """Verify color_rotation_idx cycles correctly (0, 1, 2)."""
        df_data = controlled_dataset_small['df_data']
        
        # Check that color_rotation_idx exists and has correct values
        assert 'color_rotation_idx' in df_data.columns, "Missing color_rotation_idx column"
        
        unique_rotations = sorted(df_data['color_rotation_idx'].unique())
        expected_rotations = [0, 1, 2]
        
        assert unique_rotations == expected_rotations, \
            f"Color rotation indices should be [0, 1, 2], got {unique_rotations}"
        
        # Verify each base×variant combination has all 3 rotations
        grouped = df_data.groupby(['base_sequence_idx', 'variant'])
        for (base_idx, variant), group_df in grouped:
            group_rotations = sorted(group_df['color_rotation_idx'].unique())
            assert group_rotations == expected_rotations, \
                f"Base {base_idx}, variant {variant} missing color rotations: got {group_rotations}"
    
    def test_velocity_consistency_across_color_rotations(self, controlled_dataset_small):
        """Verify velocity changes are identical for all color rotations of same base×variant."""
        targets = controlled_dataset_small['targets']
        df_data = controlled_dataset_small['df_data']
        
        # Group by base_sequence_idx and variant
        grouped = df_data.groupby(['base_sequence_idx', 'variant'])
        
        for (base_idx, variant), group_df in grouped:
            if len(group_df) < 2:
                continue
            
            # Get indices for this group
            trial_indices = group_df.index.tolist()
            
            # Extract velocity change vectors (indices 5 and 6 in targets)
            # change_vectors[:, :, 0] = bounce velocity changes
            # change_vectors[:, :, 1] = random velocity changes
            reference_vel_changes = targets[trial_indices[0], :, 5:7]  # (T, 2)
            
            # Compare all trials in this group
            for trial_idx in trial_indices[1:]:
                trial_vel_changes = targets[trial_idx, :, 5:7]
                
                np.testing.assert_array_equal(
                    reference_vel_changes,
                    trial_vel_changes,
                    err_msg=f"Velocity changes differ for base {base_idx}, variant {variant}: "
                            f"trial {trial_indices[0]} vs trial {trial_idx}"
                )
    
    def test_base_sequence_distribution(self, controlled_dataset_small):
        """Verify each base sequence is used correct number of times."""
        df_data = controlled_dataset_small['df_data']
        metadata = controlled_dataset_small['metadata']
        
        # Count usage of each base sequence
        base_counts = df_data['base_sequence_idx'].value_counts()
        
        # Expected usage per base
        num_variants = len(metadata['controlled_parameters']['variants'])
        num_colors = metadata['controlled_parameters']['num_colors']
        expected_per_base = num_variants * num_colors
        
        # Each base should be used exactly N_variants × N_colors times
        for base_idx, count in base_counts.items():
            assert count == expected_per_base, \
                f"Base sequence {base_idx} used {count} times, expected {expected_per_base}"
    
    def test_sample_position_matches_target(self, controlled_dataset_small):
        """Verify sample positions match target positions."""
        samples = controlled_dataset_small['samples']
        targets = controlled_dataset_small['targets']
        
        # Positions are first 2 channels in both samples and targets
        sample_positions = samples[:, :, :2]
        target_positions = targets[:, :, :2]
        
        # Should be exactly equal
        np.testing.assert_array_equal(
            sample_positions,
            target_positions,
            err_msg="Sample positions should exactly match target positions"
        )
    
    def test_rgb_color_values_valid(self, controlled_dataset_small):
        """Verify RGB values are valid (0 or 255 for pure colors)."""
        samples = controlled_dataset_small['samples']
        targets = controlled_dataset_small['targets']
        
        # Colors in samples are last 3 channels
        sample_colors = samples[:, :, 2:5]
        target_colors = targets[:, :, 2:5]
        
        # For controlled dataset, targets have pure colors (0 or 255)
        # Samples may have intermediate values due to masking
        unique_target_values = np.unique(target_colors)
        
        # Target colors should be pure
        for val in unique_target_values:
            assert np.isclose(val, 0) or np.isclose(val, 255), \
                f"Target color value {val} is not 0 or 255"
        
        # Sample colors should be in valid range [0, 255]
        assert np.all(sample_colors >= 0), "Sample colors should be >= 0"
        assert np.all(sample_colors <= 255), "Sample colors should be <= 255"
    
    def test_dataframe_column_completeness(self, controlled_dataset_small):
        """Verify dataframe has all required columns."""
        df_data = controlled_dataset_small['df_data']
        
        required_columns = [
            'idx', 'trial', 'variant', 'base_sequence_idx', 'color_rotation_idx',
            'Final Color', 'PCCNVC', 'PCCOVC', 'PVC',
            'Hazard Rate', 'Contingency',
            'Bounces', 'Random Bounces', 'Color Change Bounce', 'Color Change Random',
        ]
        
        missing_columns = set(required_columns) - set(df_data.columns)
        assert len(missing_columns) == 0, \
            f"Missing required columns: {missing_columns}"
    
    def test_multiple_variants(self, multiple_variants_dataset):
        """Test dataset generation with multiple variants."""
        df_data = multiple_variants_dataset['df_data']
        targets = multiple_variants_dataset['targets']
        
        # Verify we have both variants
        unique_variants = df_data['variant'].unique()
        assert set(unique_variants) == {'no_change', 'test_variant'}, \
            f"Expected both variants, got {unique_variants}"
        
        # Verify trial count: 4 base × 2 variants × 3 colors = 24
        assert len(df_data) == 24, f"Expected 24 trials, got {len(df_data)}"
        
        # Verify position consistency across variants
        for base_idx in range(4):
            base_trials = df_data[df_data['base_sequence_idx'] == base_idx]
            trial_indices = base_trials.index.tolist()
            
            if len(trial_indices) >= 2:
                ref_positions = targets[trial_indices[0], :, :2]
                for idx in trial_indices[1:]:
                    trial_positions = targets[idx, :, :2]
                    np.testing.assert_array_equal(ref_positions, trial_positions)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
