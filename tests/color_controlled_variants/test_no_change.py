"""Tests specific to the no_change variant of controlled dataset.

These tests validate the behavior of the no_change variant specifically,
ensuring that no color changes occur throughout trials.
"""
import pytest
import numpy as np
import pandas as pd

from bouncing_ball_task.color_controlled_bouncing_ball.dataset import generate_color_controlled_dataset
from bouncing_ball_task.color_controlled_bouncing_ball.no_change import generate_no_change_trials
from .base_variant_tests import BaseVariantTests


@pytest.mark.color_controlled_dataset
@pytest.mark.variant
@pytest.mark.no_change
class TestNoChangeVariant(BaseVariantTests):
    """Test the no_change variant implementation."""

    variant_functions = {'no_change': generate_no_change_trials}

    @pytest.fixture
    def variant_dataset(self, color_controlled_dataset_small):
        """Provide variant dataset for base class tests."""
        return color_controlled_dataset_small
    
    def test_no_color_changes_in_target_colors(self, color_controlled_dataset_small):
        """Verify that target color values remain constant across each trial."""
        targets = color_controlled_dataset_small['targets']
        df_data = color_controlled_dataset_small['df_data']
        
        # Check each trial - use targets to avoid masking effects
        for i, target in enumerate(targets):
            # Extract color channels from targets (indices 2:5)
            colors = target[:, 2:5]
            
            # Get initial color
            initial_color = colors[0]
            
            # Check that color remains constant throughout
            for t in range(1, len(colors)):
                np.testing.assert_allclose(
                    colors[t], initial_color,
                    err_msg=f"Color changed at timestep {t} in trial {i}"
                )
    
    def test_no_color_changes_in_targets(self, color_controlled_dataset_small):
        """Verify that color change indicators are all zero in targets."""
        targets = color_controlled_dataset_small['targets']
        
        # Change vectors are in indices 5:9
        # Specifically, color changes are at indices 7 and 8
        for i, target in enumerate(targets):
            color_change_bounce = target[:, 7]  # Color change on bounce
            color_change_random = target[:, 8]  # Random color change
            
            assert np.all(color_change_bounce == 0), \
                f"Color change on bounce detected in trial {i}"
            assert np.all(color_change_random == 0), \
                f"Random color change detected in trial {i}"
    
    def test_parameter_values_in_dataframe(self, color_controlled_dataset_small):
        """Verify that color change parameters are all zero."""
        df_data = color_controlled_dataset_small['df_data']
        
        # Check probability parameters
        assert np.all(df_data['PCCNVC'] == 0.0), \
            "PCCNVC (probability color change no velocity change) should be 0"
        assert np.all(df_data['PCCOVC'] == 0.0), \
            "PCCOVC (probability color change on velocity change) should be 0"
        
        # Check effective parameters if present
        if 'PCCNVC_effective' in df_data.columns:
            assert np.all(df_data['PCCNVC_effective'] == 0.0), \
                "Effective PCCNVC should be 0"
        if 'PCCOVC_effective' in df_data.columns:
            assert np.all(df_data['PCCOVC_effective'] == 0.0), \
                "Effective PCCOVC should be 0"
    
    def test_color_change_counts(self, color_controlled_dataset_small):
        """Verify that color change counts are zero."""
        df_data = color_controlled_dataset_small['df_data']
        
        # Check color change counts
        assert np.all(df_data['Color Change Bounce'] == 0), \
            "Color change bounce count should be 0"
        assert np.all(df_data['Color Change Random'] == 0), \
            "Color change random count should be 0"
    
    def test_hazard_rate_contingency_labels(self, color_controlled_dataset_small):
        """Hazard Rate / Contingency use the REAL task's vocabulary.

        compute_effective_stats bins them from the unique PCCNVC / PCCOVC values
        (HDS:573-597); a single-variant color-controlled set has one value each,
        which maps to the lowest category. The labels are emitted for FORMAT
        parity — their statistical informativeness is an explicit non-goal.
        """
        df_data = color_controlled_dataset_small['df_data']

        assert set(df_data['Hazard Rate'].cat.categories) == {'Low', 'High'}
        assert set(df_data['Contingency'].cat.categories) == {'Low', 'Medium', 'High'}
        assert np.all(df_data['Hazard Rate'] == 'Low')
        assert np.all(df_data['Contingency'] == 'Low')
        
        # Check they are categorical type
        assert isinstance(df_data['Hazard Rate'].dtype, pd.CategoricalDtype), \
            "Hazard Rate should be categorical"
        assert isinstance(df_data['Contingency'].dtype, pd.CategoricalDtype), \
            "Contingency should be categorical"
    
    def test_variant_metadata(self, color_controlled_dataset_small):
        """Verify that no_change variant metadata is correct."""
        metadata = color_controlled_dataset_small['metadata']
        
        # Check that no_change is in the metadata
        assert 'no_change' in metadata, "no_change variant should be in metadata"
        
        no_change_meta = metadata['no_change']
        assert 'variant_metadata' in no_change_meta, \
            "no_change should have variant_metadata"
        
        variant_meta = no_change_meta['variant_metadata']
        assert variant_meta.get('variant_name') == 'no_change', \
            "Variant name should be 'no_change'"
        assert variant_meta.get('change_parameters', {}).get('allow_color_changes') == False, \
            "allow_color_changes should be False"
        # Authored-event bookkeeping derived by the generator
        assert variant_meta['authored_changes'] == {
            'color_change_bounce': 0, 'color_change_random': 0,
        }
        # With frame 0 natively zero (initial_timestep_is_changepoint=False) and
        # PVC=0, the metadata claim random_changes == 0 is now literally true
        assert variant_meta['expected_changes']['random_changes'] == 0
        assert np.all(color_controlled_dataset_small['df_data']['Random Bounces'] == 0)

    def test_no_change_returns_zero_color_events(self):
        """The variant's authored events are all-zero (N, T, 2)."""
        positions = np.zeros((3, 20, 2))
        bounce_frames = np.zeros((3, 20), dtype=bool)
        bounce_frames[0, 5] = True
        events = generate_no_change_trials(positions, bounce_frames, {})
        assert events.shape == (3, 20, 2)
        assert np.all(events == 0)
    
    def test_velocity_changes_allowed(self, color_controlled_dataset_small):
        """Verify that velocity changes are still allowed (only color changes disabled)."""
        df_data = color_controlled_dataset_small['df_data']
        targets = color_controlled_dataset_small['targets']
        
        # Check that some bounces occur (wall bounces happen regardless of PVC)
        total_bounces = df_data['Bounces'].sum()
        assert total_bounces > 0, \
            "There should be some wall bounces in the dataset"

        # PVC=0: the random-velocity ledger (ch6) is identically zero
        assert np.all(targets[:, :, 6] == 0), "ch6 (random velocity change) must be 0 under PVC=0"
        
        # Verify bounce indicators in targets
        bounce_occurred = False
        for target in targets:
            bounce_indicators = target[:, 5]  # Bounce velocity change
            if np.any(bounce_indicators > 0):
                bounce_occurred = True
                break
        
        assert bounce_occurred, "At least one wall bounce should occur in the dataset"
    
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces identical results."""
        color_controlled_params = {
            'num_base_sequences': 3,
            'seed': 54321,
            'duration': 50,
            'variable_length': False,
        }
        
        task_params = {
            'sequence_length': 50,
            'size_frame': (256, 256),
            'ball_radius': 10,
        }
        
        # Generate two datasets with same seed
        np.random.seed(color_controlled_params['seed'])
        result1 = generate_color_controlled_dataset(
            color_controlled_params.copy(),
            task_params.copy(),
            shuffle=False,
            dict_trial_type_generation_funcs={'no_change': generate_no_change_trials}
        )
        
        np.random.seed(color_controlled_params['seed'])
        result2 = generate_color_controlled_dataset(
            color_controlled_params.copy(),
            task_params.copy(),
            shuffle=False,
            dict_trial_type_generation_funcs={'no_change': generate_no_change_trials}
        )
        
        # Compare key outputs
        targets1 = result1[3]  # targets
        targets2 = result2[3]  # targets
        
        np.testing.assert_array_equal(targets1, targets2,
                                      "Same seed should produce identical targets")
        
        # Compare dataframes
        df1 = result1[4]  # df_data
        df2 = result2[4]  # df_data
        
        # Check key columns are identical
        for col in ['base_sequence_idx', 'color_rotation_idx', 'Final Color']:
            np.testing.assert_array_equal(df1[col].values, df2[col].values,
                                          f"Column {col} should be identical with same seed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
