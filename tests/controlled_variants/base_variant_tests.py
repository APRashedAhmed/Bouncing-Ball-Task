"""Base test class for common variant tests.

This module contains tests that should pass for all controlled dataset variants.
Specific variant test classes should inherit from BaseVariantTests.
"""
import pytest
import numpy as np
import pandas as pd


class BaseVariantTests:
    """Common tests that all controlled dataset variants should pass."""
    
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
        num_base = variant_dataset['controlled_params']['num_base_sequences']
        
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
    
    def test_change_vector_shape(self, variant_dataset):
        """Verify change vectors have correct shape and structure."""
        targets = variant_dataset['targets']
        
        # Change vectors should be in indices 5:9
        for i, target in enumerate(targets):
            change_vector = target[:, 5:9]
            assert change_vector.shape[1] == 4, \
                f"Change vector should have 4 channels, got {change_vector.shape[1]}"
            
            # Verify all values are finite
            assert np.all(np.isfinite(change_vector)), \
                f"Change vector contains non-finite values in trial {i}"
    
    def test_dataframe_completeness(self, variant_dataset):
        """Verify dataframe has all required columns."""
        df_data = variant_dataset['df_data']
        
        required_columns = [
            'idx', 'length', 'trial', 'variant',
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
        num_base = variant_dataset['controlled_params']['num_base_sequences']
        
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