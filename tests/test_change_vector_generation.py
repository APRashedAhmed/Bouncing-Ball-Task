"""Tests for change vector generation and manipulation.

These tests validate the core change vector approach used in the controlled dataset.
"""
import pytest
import numpy as np

from bouncing_ball_task.controlled_bouncing_ball.dataset import generate_controlled_dataset
from bouncing_ball_task.controlled_bouncing_ball.no_change import generate_no_change_trials


@pytest.mark.controlled_dataset
class TestChangeVectorGeneration:
    """Test change vector generation and manipulation."""
    
    def test_change_vector_shape(self, controlled_dataset_small):
        """Verify change vectors have correct shape."""
        targets = controlled_dataset_small['targets']
        
        # Change vectors should be in indices 5:9 of targets
        # Shape should be (N, T, 4) where:
        # - N is number of trials
        # - T is sequence length
        # - 4 is [bounce_velocity, random_velocity, color_bounce, color_random]
        
        for i, target in enumerate(targets):
            change_vector = target[:, 5:9]
            assert change_vector.shape[1] == 4, \
                f"Change vector should have 4 channels, got {change_vector.shape[1]}"
    
    def test_base_sequence_reuse(self, controlled_dataset_small):
        """Verify that base sequences are reused across color rotations."""
        df_data = controlled_dataset_small['df_data']
        targets = controlled_dataset_small['targets']
        
        # Get unique base sequences
        unique_bases = df_data['base_sequence_idx'].unique()
        
        for base_idx in unique_bases:
            base_trials = df_data[df_data['base_sequence_idx'] == base_idx]
            
            # Should have exactly 3 trials per base (3 color rotations)
            assert len(base_trials) == 3, \
                f"Base sequence {base_idx} should have 3 trials, got {len(base_trials)}"
            
            # Get trial indices
            trial_indices = base_trials.index.tolist()
            
            # Extract velocity change vectors (indices 5 and 6)
            ref_velocity_changes = targets[trial_indices[0], :, 5:7]
            
            # All trials from same base should have identical velocity changes
            for idx in trial_indices[1:]:
                trial_velocity_changes = targets[idx, :, 5:7]
                np.testing.assert_array_equal(
                    ref_velocity_changes,
                    trial_velocity_changes,
                    err_msg=f"Velocity changes differ for base {base_idx}"
                )
    
    def test_color_change_zeroing(self, controlled_dataset_small):
        """Verify that color change channels are zeroed in base generation."""
        targets = controlled_dataset_small['targets']
        
        # Color change indicators are at indices 7 and 8
        for i, target in enumerate(targets):
            color_changes = target[:, 7:9]
            
            assert np.all(color_changes == 0), \
                f"Color changes should be zeroed in trial {i}"
    
    def test_cumulative_color_generation(self, controlled_dataset_small):
        """Test the cumulative sum approach for color generation."""
        targets = controlled_dataset_small['targets']
        samples = controlled_dataset_small['samples']
        
        # For no_change variant, colors should remain constant
        # since color change indicators are all zero
        for i in range(len(samples)):
            sample_colors = samples[i, :, 2:5]  # RGB values
            target_colors = targets[i, :, 2:5]

            # Sample colors must stay within the valid RGB range
            assert np.all((sample_colors >= 0) & (sample_colors <= 255)), \
                f"Sample colors out of [0, 255] range in trial {i}"

            # Initial color
            initial_color = target_colors[0]
            
            # All colors should match initial color (no changes)
            for t in range(1, len(target_colors)):
                np.testing.assert_allclose(
                    target_colors[t], initial_color,
                    err_msg=f"Target color changed at t={t} in trial {i}"
                )
    
    def test_position_extraction(self, controlled_dataset_small):
        """Verify that positions are correctly extracted from targets."""
        targets = controlled_dataset_small['targets']
        samples = controlled_dataset_small['samples']
        
        # Positions should be first 2 channels in both samples and targets
        for i in range(len(samples)):
            sample_positions = samples[i, :, :2]
            target_positions = targets[i, :, :2]
            
            np.testing.assert_array_equal(
                sample_positions,
                target_positions,
                err_msg=f"Positions don't match between samples and targets in trial {i}"
            )
    
    def test_variant_concatenation(self, multiple_variants_dataset):
        """Test that variants are properly concatenated."""
        df_data = multiple_variants_dataset['df_data']
        
        # Check variant distribution
        variants = df_data['variant'].values
        num_base = multiple_variants_dataset['controlled_params']['num_base_sequences']
        num_colors = 3
        
        # The pattern is: for each color rotation, we have all variants
        # So for 2 variants and 3 colors with 4 base sequences:
        # Color 0: no_change (4x), test_variant (4x)
        # Color 1: no_change (4x), test_variant (4x)
        # Color 2: no_change (4x), test_variant (4x)
        
        # Check we have the right total number
        expected_total = num_base * 2 * num_colors  # 2 variants
        assert len(variants) == expected_total, \
            f"Expected {expected_total} trials, got {len(variants)}"
        
        # Check equal distribution of variants
        no_change_count = np.sum(variants == 'no_change')
        test_variant_count = np.sum(variants == 'test_variant')
        
        assert no_change_count == num_base * num_colors, \
            f"Expected {num_base * num_colors} no_change trials, got {no_change_count}"
        assert test_variant_count == num_base * num_colors, \
            f"Expected {num_base * num_colors} test_variant trials, got {test_variant_count}"
    
    def test_color_rotation_pattern(self, controlled_dataset_small):
        """Verify the pattern of color rotations."""
        df_data = controlled_dataset_small['df_data']
        
        # Group by color rotation
        for color_idx in [0, 1, 2]:
            color_trials = df_data[df_data['color_rotation_idx'] == color_idx]
            
            # Each color rotation should have all base sequences
            base_sequences = sorted(color_trials['base_sequence_idx'].unique())
            expected_bases = list(range(controlled_dataset_small['controlled_params']['num_base_sequences']))
            
            assert base_sequences == expected_bases, \
                f"Color rotation {color_idx} missing base sequences"
    
    def test_final_target_shape(self, controlled_dataset_small):
        """Verify final target array has correct shape."""
        targets = controlled_dataset_small['targets']
        df_data = controlled_dataset_small['df_data']
        
        num_trials = len(df_data)
        sequence_length = controlled_dataset_small['task_params']['sequence_length']
        
        expected_shape = (num_trials, sequence_length, 9)
        assert targets.shape == expected_shape, \
            f"Targets shape {targets.shape} doesn't match expected {expected_shape}"
        
        # Verify the 9 channels are:
        # [x, y, r, g, b, bounce_vc, random_vc, color_bounce, color_random]
        assert targets.shape[2] == 9, "Targets should have 9 channels"
    
    def test_base_task_properties(self, controlled_dataset_small):
        """Verify base task properties are preserved."""
        task = controlled_dataset_small['task']
        task_params = controlled_dataset_small['task_params']
        
        # Check task has expected properties
        assert task.sequence_length == task_params['sequence_length']
        assert np.array_equal(task.size_frame, task_params['size_frame'])
        assert task.ball_radius == task_params['ball_radius']
        
        # Check samples and targets match
        assert len(task.samples) == len(task.targets)
        assert task.samples.shape[1] == task.sequence_length
        assert task.targets.shape[1] == task.sequence_length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
