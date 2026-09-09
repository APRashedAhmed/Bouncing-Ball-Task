"""Base test class for common variant tests.

This module contains tests that should pass for all controlled dataset variants.
Specific variant test classes should inherit from BaseVariantTests.
"""
import pytest
import numpy as np
import pandas as pd


class BaseVariantTests:
    """Common tests that all controlled dataset variants should pass."""

    #: name -> color-authoring function for every variant in ``variant_dataset``
    variant_functions: dict = {}

    @pytest.fixture
    def variant_dataset(self):
        """Override this fixture in variant-specific test classes."""
        raise NotImplementedError("Subclasses must provide variant_dataset fixture")
    
    def test_position_consistency_across_color_rotations(self, variant_dataset):
        """Verify that positions are identical across color rotations for same base sequence."""
        df_data = variant_dataset['df_data']
        targets = variant_dataset['targets']
        
        # Group by base sequence
        for base_idx in df_data['base_sequence_idx'].unique():
            base_trials = df_data[df_data['base_sequence_idx'] == base_idx]
            trial_indices = base_trials.index.tolist()
            
            # Get reference positions from first trial
            ref_positions = targets[trial_indices[0], :, :2]
            
            # Compare all other trials from same base
            for idx in trial_indices[1:]:
                trial_positions = targets[idx, :, :2]
                np.testing.assert_array_equal(
                    ref_positions, 
                    trial_positions,
                    err_msg=f"Positions differ for base sequence {base_idx}"
                )
    
    def test_metadata_structure(self, variant_dataset):
        """Verify that variant provides proper metadata structure."""
        metadata = variant_dataset['metadata']
        
        # Check basic metadata fields
        assert 'controlled_parameters' in metadata
        assert 'task_parameters' in metadata
        assert 'controlled_dataset_parameters' in metadata
        
        # Check variant-specific metadata exists
        # (This will be the variant name like 'no_change', 'sudden_change', etc.)
        variant_names = metadata['controlled_parameters']['variants']
        for variant_name in variant_names:
            assert variant_name in metadata, f"Metadata missing for variant {variant_name}"
            variant_meta = metadata[variant_name]
            assert 'variant_metadata' in variant_meta
            assert 'num_trials' in variant_meta
    
    def test_three_color_rotation_pattern(self, variant_dataset):
        """Verify that each base sequence has exactly 3 color rotations."""
        df_data = variant_dataset['df_data']
        num_base = variant_dataset['color_controlled_params']['num_base_sequences']
        
        # Check each base sequence
        for base_idx in range(num_base):
            base_trials = df_data[df_data['base_sequence_idx'] == base_idx]
            
            # Should have exactly 3 color rotations
            color_rotations = sorted(base_trials['color_rotation_idx'].unique())
            assert color_rotations == [0, 1, 2], \
                f"Base sequence {base_idx} should have color rotations [0, 1, 2]"
            
            # Should have all three colors
            final_colors = set(base_trials['Final Color'].unique())
            assert final_colors == {'red', 'green', 'blue'}, \
                f"Base sequence {base_idx} should have all 3 colors"
    
    def test_variant_authors_color_channels_only(self, variant_dataset):
        """Verify the color-only authoring contract for every registered variant.

        ``variant(positions, bounce_frames, params) -> (N, T, 2)`` — the two
        authored color channels [cc_bounce (7), cc_random (8)]; velocity
        channels are never exposed for writing.
        """
        from bouncing_ball_task.color_controlled_bouncing_ball import dataset as cds

        targets = variant_dataset['targets']
        df_data = variant_dataset['df_data']
        metadata = variant_dataset['metadata']

        # One base sequence per base index (any rotation): the read-only inputs
        first_rows = df_data.drop_duplicates('base_sequence_idx').sort_values('base_sequence_idx')
        base_targets = targets[first_rows.index.values]
        positions = base_targets[:, :, :2]
        bounce_frames = base_targets[:, :, 5].astype(bool)
        params = {
            'controlled_dataset_parameters': variant_dataset['color_controlled_params'],
            'task_parameters': metadata['task_parameters'],
            'initial_position': metadata['controlled_parameters']['initial_position'],
            'initial_velocity': metadata['controlled_parameters']['initial_velocity'],
            'control_end': metadata['controlled_parameters']['control_end'],
        }

        for variant_name in metadata['controlled_parameters']['variants']:
            variant_func = self.variant_functions[variant_name]
            color_events = variant_func(positions.copy(), bounce_frames.copy(), params)
            assert isinstance(color_events, np.ndarray)
            assert color_events.shape == positions.shape[:2] + (2,), \
                f"Variant '{variant_name}' must return (N, T, 2), got {color_events.shape}"
            assert np.all(np.isin(color_events, (0, 1))), \
                f"Variant '{variant_name}' color events must be binary"
            # The runtime invariant must accept what the variant authored
            cds.validate_color_events(color_events, bounce_frames, variant_name)

        # Velocity channels in the final targets are the derived ledger only
        assert np.all(np.isfinite(targets[:, :, 5:9]))
        assert targets.shape[2] == 9
    
    def test_dataframe_completeness(self, variant_dataset):
        """Verify dataframe has all required columns."""
        df_data = variant_dataset['df_data']
        
        required_columns = [
            'idx_trial', 'length', 'trial', 'variant',
            'base_sequence_idx', 'color_rotation_idx',
            'Final Color', 'Bounces', 'Random Bounces',
            'Color Change Bounce', 'Color Change Random',
            'PCCNVC', 'PCCOVC', 'PVC',
            'Hazard Rate', 'Contingency'
        ]
        
        for col in required_columns:
            assert col in df_data.columns, f"Missing required column: {col}"
    
    def test_trial_count_formula(self, variant_dataset):
        """Verify total trial count matches formula: base_sequences × variants × 3."""
        df_data = variant_dataset['df_data']
        num_base = variant_dataset['color_controlled_params']['num_base_sequences']
        
        # Count unique variants
        num_variants = df_data['variant'].nunique()
        num_colors = 3  # Always 3 color rotations
        
        expected_trials = num_base * num_variants * num_colors
        actual_trials = len(df_data)
        
        assert actual_trials == expected_trials, \
            f"Expected {expected_trials} trials, got {actual_trials}"
    
    def test_reproducibility_with_seed(self, variant_dataset):
        """Test that same seed produces identical results for the variant."""
        # This test should be implemented by each variant
        # as they need to generate their specific variant
        pytest.skip("Implement in variant-specific test class")
