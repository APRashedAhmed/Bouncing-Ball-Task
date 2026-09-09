"""End-product tests for controlled dataset generation.

These tests validate the final output of the controlled dataset generation,
focusing on critical properties like position consistency and color uniqueness.
"""
import pytest
import numpy as np
import pandas as pd
from collections import defaultdict

from bouncing_ball_task.bouncing_ball import BouncingBallTask
from bouncing_ball_task.constants import DEFAULT_COLORS
from bouncing_ball_task.human_bouncing_ball import defaults as human_defaults
from bouncing_ball_task.color_controlled_bouncing_ball.dataset import generate_color_controlled_dataset
from bouncing_ball_task.color_controlled_bouncing_ball.no_change import generate_no_change_trials


def _recompute_last_visible_color_idx(trajectory, length, min_length, task):
    """Independently recompute `last_visible_color_idx` from positions alone.

    Mirrors the OUTER band definition of ``taskutils.last_visible_color``
    (taskutils.py:74-78, ``tol=0``) over the last ``min_length`` frames, plus the
    ``length - min_length`` offset applied at HDS:376.
    """
    x = np.asarray(trajectory)[-min_length:, 0]
    visible = (x < task.mask_start - task.ball_radius) | (x > task.mask_end + task.ball_radius)
    assert visible.any(), "test premise: the ball is visible at some point"
    return int(np.flatnonzero(visible)[-1]) + int(length) - int(min_length)


def _generate(color_controlled_params, task_params, **kwargs):
    kwargs.setdefault('shuffle', False)
    kwargs.setdefault('dict_trial_type_generation_funcs', {'no_change': generate_no_change_trials})
    return generate_color_controlled_dataset(color_controlled_params, task_params, **kwargs)


@pytest.mark.color_controlled_dataset
class TestColorControlledDatasetEndProduct:
    """Test the end product of controlled dataset generation."""
    
    def test_position_consistency_across_variants(self, color_controlled_dataset_small):
        """CRITICAL: Verify all variants from same base have identical positions at all timesteps."""
        targets = color_controlled_dataset_small['targets']
        df_data = color_controlled_dataset_small['df_data']
        
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
    
    def test_color_uniqueness_per_variant(self, color_controlled_dataset_small):
        """CRITICAL: Verify each variant produces all 3 unique final colors."""
        df_data = color_controlled_dataset_small['df_data']
        
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
    
    def test_trial_count_formula(self, color_controlled_dataset_small):
        """Verify total trials = N_base × N_variants × N_colors."""
        df_data = color_controlled_dataset_small['df_data']
        metadata = color_controlled_dataset_small['metadata']
        color_controlled_params = color_controlled_dataset_small['color_controlled_params']
        
        # Extract counts
        num_base = color_controlled_params['num_base_sequences']
        num_variants = len(metadata['controlled_parameters']['variants'])
        num_colors = metadata['controlled_parameters']['num_colors']
        
        # Calculate expected total
        expected_total = num_base * num_variants * num_colors
        actual_total = len(df_data)
        
        assert actual_total == expected_total, \
            f"Trial count mismatch: expected {num_base}×{num_variants}×{num_colors}={expected_total}, " \
            f"got {actual_total}"
    
    def test_color_rotation_indices(self, color_controlled_dataset_small):
        """Verify color_rotation_idx cycles correctly (0, 1, 2)."""
        df_data = color_controlled_dataset_small['df_data']
        
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
    
    def test_velocity_consistency_across_color_rotations(self, color_controlled_dataset_small):
        """Verify velocity changes are identical for all color rotations of same base×variant."""
        targets = color_controlled_dataset_small['targets']
        df_data = color_controlled_dataset_small['df_data']
        
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
    
    def test_base_sequence_distribution(self, color_controlled_dataset_small):
        """Verify each base sequence is used correct number of times."""
        df_data = color_controlled_dataset_small['df_data']
        metadata = color_controlled_dataset_small['metadata']
        
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
    
    def test_sample_position_matches_target(self, color_controlled_dataset_small):
        """Verify sample positions match target positions."""
        samples = color_controlled_dataset_small['samples']
        targets = color_controlled_dataset_small['targets']
        
        # Positions are first 2 channels in both samples and targets
        sample_positions = samples[:, :, :2]
        target_positions = targets[:, :, :2]
        
        # Should be exactly equal
        np.testing.assert_array_equal(
            sample_positions,
            target_positions,
            err_msg="Sample positions should exactly match target positions"
        )
    
    def test_rgb_color_values_valid(self, color_controlled_dataset_small):
        """Verify RGB values are valid (0 or 255 for pure colors)."""
        samples = color_controlled_dataset_small['samples']
        targets = color_controlled_dataset_small['targets']
        
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
    
    def test_dataframe_column_completeness(self, color_controlled_dataset_small):
        """Verify dataframe has all required columns."""
        df_data = color_controlled_dataset_small['df_data']
        
        required_columns = [
            'idx_trial', 'trial', 'variant', 'base_sequence_idx', 'color_rotation_idx',
            'control_end', 'Final Color', 'PCCNVC', 'PCCOVC', 'PVC',
            'Hazard Rate', 'Contingency',
            'Bounces', 'Random Bounces', 'Color Change Bounce', 'Color Change Random',
            # Real-task schema (P5 parity)
            'length', 'length_ms', 'Dataset Block', 'Dataset Block Video',
            'Final X Position', 'Final Y Position',
            'Final X Velocity', 'Final Y Velocity',
            'last_visible_color_idx', 'last_visible_color_idx_cent',
            'last_visible_color_idx_inner', 'last_visible_color',
            'color_entered', 'color_next', 'color_after_next', 'correct_response',
            'idx_time', 'idx_position', 'idx_velocity_y', 'idx_x_position',
            'side_left_right', 'side_top_bottom',
            'Bounces observable', 'Color Change Bounce observable',
            'PCCNVC_effective', 'PCCNVC_effective_ps',
            'PCCOVC_effective', 'PVC_effective', 'PVC_effective_ps',
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


@pytest.mark.color_controlled_dataset
class TestControlledTrajectory:
    """P3: keep (control the start) vs reverse (control the end)."""

    PARAMS = {'num_base_sequences': 4, 'seed': 99,
              'initial_position': [[40.0, 30.0], [40.0, 30.0], [210.0, 200.0], [210.0, 200.0]],
              'initial_velocity': [[-25.0, 20.0], [-25.0, 20.0], [25.0, -30.0], [25.0, -30.0]],
              'control_end': [False, True, False, True]}
    TASK = {'sequence_length': 80}

    @pytest.fixture
    def generated(self):
        return _generate(dict(self.PARAMS), dict(self.TASK))

    def test_static_trials_start_and_reversed_trials_end_at_requested_position(self, generated):
        _, _, _, targets, df, _ = generated
        requested = np.asarray(self.PARAMS['initial_position'])
        for _, row in df.iterrows():
            trajectory = targets[row['idx_trial'], :, :2]
            base = row['base_sequence_idx']
            if row['control_end']:
                np.testing.assert_allclose(trajectory[-1], requested[base], atol=1e-3,
                                           err_msg=f"reversed trial {row['idx_trial']} must END at the requested position")
                assert not np.allclose(trajectory[0], requested[base], atol=1e-3)
            else:
                np.testing.assert_allclose(trajectory[0], requested[base], atol=1e-3,
                                           err_msg=f"static trial {row['idx_trial']} must START at the requested position")
                assert not np.allclose(trajectory[-1], requested[base], atol=1e-3)

    def test_reversed_trial_is_the_time_reversal_of_its_static_twin(self, generated):
        """Same init conditions: the reverse trajectory is the static one flipped."""
        _, _, _, targets, df, _ = generated
        rot0 = df[df['color_rotation_idx'] == 0].set_index('base_sequence_idx')
        for static_base, reverse_base in ((0, 1), (2, 3)):
            static = targets[rot0.loc[static_base, 'idx_trial'], :, :2]
            reverse = targets[rot0.loc[reverse_base, 'idx_trial'], :, :2]
            np.testing.assert_allclose(reverse, static[::-1], atol=1e-3)

    def test_final_position_and_velocity_values(self, generated):
        """P5 values, not just presence: `Final X/Y Position` is the last frame of
        the trajectory, and `Final X/Y Velocity` follows the two-branch convention
        (CC:806-812) — reverse trials `-initial_velocity`, static trials the
        trailing finite difference `(p[-1] - p[-2]) / dt`."""
        task, _, _, targets, df, _ = generated
        dt = task.dt
        initial_velocity = np.asarray(self.PARAMS['initial_velocity'])
        saw_static = False
        branches_distinguishable = False
        for _, row in df.iterrows():
            trajectory = targets[row['idx_trial'], :, :2]
            base = int(row['base_sequence_idx'])
            np.testing.assert_allclose(
                [row['Final X Position'], row['Final Y Position']],
                trajectory[-1], atol=1e-3, err_msg=f"trial {row['idx_trial']}")

            final_velocity = np.array([row['Final X Velocity'], row['Final Y Velocity']])
            finite_difference = (trajectory[-1] - trajectory[-2]) / dt
            # The finite difference pins the dt factor in BOTH branches
            np.testing.assert_allclose(final_velocity, finite_difference, atol=1e-3,
                                       err_msg=f"trial {row['idx_trial']}")
            if row['control_end']:
                np.testing.assert_allclose(final_velocity, -initial_velocity[base], atol=1e-3)
            else:
                np.testing.assert_allclose(final_velocity, finite_difference, atol=1e-3)
                saw_static = True
                # A static trial whose finite difference differs from
                # -initial_velocity is what makes a swapped branch detectable
                if not np.allclose(final_velocity, -initial_velocity[base], atol=1e-3):
                    branches_distinguishable = True
        assert saw_static, "test premise: the fixture must contain static trials"
        assert branches_distinguishable, \
            "test premise: at least one static trial must disagree with -initial_velocity"

    def test_control_end_column_and_metadata_bookkeeping(self, generated):
        _, _, _, _, df, meta = generated
        assert df['control_end'].dtype == bool
        expected = np.asarray(self.PARAMS['control_end'])
        np.testing.assert_array_equal(df['control_end'].values, expected[df['base_sequence_idx'].values])
        np.testing.assert_array_equal(meta['controlled_parameters']['control_end'], expected)
        np.testing.assert_array_equal(meta['controlled_parameters']['initial_position'],
                                      self.PARAMS['initial_position'])

    def test_bounce_flag_sits_one_frame_after_the_turning_point(self, generated):
        """ch5 is position-consistent in BOTH modes (P3.0)."""
        _, _, _, targets, df, _ = generated
        for _, row in df[df['color_rotation_idx'] == 0].iterrows():
            trajectory = targets[row['idx_trial'], :, :2]
            flags = np.flatnonzero(targets[row['idx_trial'], :, 5])
            assert len(flags) > 0
            for t in flags:
                before, after = trajectory[t - 1] - trajectory[t - 2], trajectory[t] - trajectory[t - 1]
                assert np.any(np.sign(before) != np.sign(after)), (row['idx_trial'], t)

    def test_control_end_defaults_to_all_false(self):
        params = {'num_base_sequences': 3, 'seed': 5}
        _, _, _, targets, df, meta = _generate(params, dict(self.TASK))
        assert not df['control_end'].any()
        np.testing.assert_allclose(targets[:3, 0, :2], meta['controlled_parameters']['initial_position'], atol=1e-3)

    def test_variable_length_requires_control_end(self):
        with pytest.raises(ValueError, match="control_end"):
            _generate({'num_base_sequences': 2, 'seed': 5, 'variable_length': True,
                       'control_end': [True, False]}, dict(self.TASK))
        # All control_end=True is allowed, and the arrays are truncated to `length`.
        # Lengths follow the human/model sampler: frames =
        # rint((Exp(exp_scale) + video_length_min_s) / duration), < 3x the
        # minimum, and the task integrates at the longest sampled length (HDS:87).
        duration, video_length_min_s = 50, 1.0
        task, samples, model_samples, targets, df, meta = _generate(
            {'num_base_sequences': 2, 'seed': 5, 'variable_length': True,
             'control_end': [True, True], 'duration': duration,
             'video_length_min_s': video_length_min_s, 'exp_scale': 0.5},
            dict(self.TASK))
        assert df['control_end'].all()
        min_f = int(round(video_length_min_s * 1000 / duration))
        assert (df['length'] >= min_f).all() and (df['length'] < 3 * min_f).all()
        assert df['length'].nunique() > 1
        assert task.sequence_length == df['length'].max() == meta['video_length_max_f']
        assert meta['task_parameters']['sequence_length'] == task.sequence_length
        requested = meta['controlled_parameters']['initial_position']
        for i, row in df.iterrows():
            length = int(row['length'])
            for array in (samples, model_samples, targets):
                assert array[i].shape[0] == length, (i, array[i].shape, length)
            # Truncation keeps the LAST `length` frames of the full arrays the
            # preset task still holds (HDS:833-835)...
            np.testing.assert_array_equal(targets[i], task.targets[i, -length:])
            np.testing.assert_array_equal(samples[i], task.samples[i, -length:])
            np.testing.assert_array_equal(model_samples[i], task.model_samples[i, -length:])
            # ...so the controlled END survives truncation
            np.testing.assert_allclose(targets[i][-1, :2],
                                       requested[row['base_sequence_idx']], atol=1e-3)


@pytest.mark.color_controlled_dataset
class TestTaskParameterAndMaskParity:
    """P5: stimulus parameters and masking are those of the real task."""

    def test_task_parameters_inherit_real_task_values(self, color_controlled_dataset_small):
        tp = color_controlled_dataset_small['metadata']['task_parameters']
        human = human_defaults.TaskParameters()
        for key in ('dt', 'color_mask_mode', 'warmup_t_no_rand_velocity_change',
                    'warmup_t_no_rand_color_change', 'min_t_color_change_after_bounce',
                    'min_t_velocity_change_after_bounce', 'transition_tol',
                    'initial_timestep_is_changepoint', 'sample_velocity_discretely'):
            assert tp[key] == getattr(human, key), key
        assert tp['dt'] == 0.1 and tp['color_mask_mode'] == 'outer'
        assert tp['initial_timestep_is_changepoint'] is False
        assert tp['probability_velocity_change'] == 0.0
        for non_task_field in ('gravity_mag', 'elasticity', 'mask_start', 'mask_end'):
            assert non_task_field not in tp
        # Final preset parameters, as the real task stores them
        assert tp['seed'] is False
        assert tp['sequence_mode'] == 'preset'
        assert tp['batch_size'] == len(color_controlled_dataset_small['df_data'])
        assert 'initial_position' not in tp and 'initial_velocity' not in tp

    def test_default_sequence_length_is_inherited(self):
        _, samples, _, _, _, _ = _generate({'num_base_sequences': 1, 'seed': 1}, {})
        assert samples.shape[1] == 600

    def test_task_parameters_round_trip_without_reseeding(self, color_controlled_dataset_small, monkeypatch):
        """(k): BouncingBallTask(**tp, samples=..., targets=...) rebuilds the task and
        never reseeds the global RNG (seed=False).

        Note: the preset constructor still CONSUMES draws (it samples an unused
        initial position / velocity / color index), so the assertion is "no
        reseed", not "RNG state unchanged".
        """
        import bouncing_ball_task.bouncing_ball as bb_module

        d = color_controlled_dataset_small
        tp = d['metadata']['task_parameters']

        def forbidden_reseed(*args, **kwargs):
            raise AssertionError("BouncingBallTask(**task_parameters) reseeded the global RNG")

        monkeypatch.setattr(bb_module.pyutils, 'set_global_seed', forbidden_reseed)
        rebuilt = BouncingBallTask(**tp, samples=d['samples'], targets=d['targets'])
        assert rebuilt.seed is False
        assert rebuilt.resolved_seed is None
        assert rebuilt.sequence_mode == 'preset'
        np.testing.assert_array_equal(rebuilt.samples, d['samples'])
        np.testing.assert_array_equal(rebuilt.targets, d['targets'])
        np.testing.assert_array_equal(rebuilt.model_samples, d['model_samples'])

    def test_mask_semantics_match_bouncing_ball_task(self):
        """(g): samples are gray in the OUTER band, model_samples restore the true
        color in the transition band (outer & ~inner), and both equal exactly what
        BouncingBallTask itself produces for the same initial conditions and color."""
        # Two trials per sub-batch: BouncingBallTask with batch_size=1 keeps
        # color_sampling='fixed' (not vectorised, BB:450) and cycles the color
        # every frame — a base-task quirk the generator sidesteps by
        # re-authoring colors, but the reference task here would trip on it.
        params = {'num_base_sequences': 4, 'seed': 77, 'control_end': [False, True, True, False]}
        task_params = {'sequence_length': 200}
        task, samples, model_samples, targets, df, meta = _generate(params, task_params)

        tp = {k: v for k, v in meta['task_parameters'].items()
              if k not in ('batch_size', 'sequence_mode', 'seed')}
        starts = meta['controlled_parameters']['initial_position']
        velocities = meta['controlled_parameters']['initial_velocity']
        control_end = meta['controlled_parameters']['control_end']

        for color_idx in range(3):
            rows = df[df['color_rotation_idx'] == color_idx].sort_values('base_sequence_idx')
            for mode, selector in (('static', ~control_end), ('reverse', control_end)):
                idx = np.flatnonzero(selector)
                reference = BouncingBallTask(
                    **tp, batch_size=len(idx), sequence_mode=mode, seed=False,
                    initial_position=starts[idx], initial_velocity=velocities[idx],
                    initial_color=[DEFAULT_COLORS[color_idx]] * len(idx),
                )
                trial_idx = rows['idx_trial'].values[idx]
                np.testing.assert_array_equal(samples[trial_idx], reference.samples)
                np.testing.assert_array_equal(model_samples[trial_idx], reference.model_samples)
                np.testing.assert_array_equal(targets[trial_idx], reference.targets)

        # Band semantics, stated directly
        x = samples[:, :, 0]
        outer = task.infer_grayzone_locations(x, mode='outer')
        inner = task.infer_grayzone_locations(x, mode='inner')
        transition = outer & ~inner
        assert transition.sum() > 0
        assert np.all(samples[outer][:, 2:] == task.mask_color)
        assert np.all(model_samples[inner][:, 2:] == task.mask_color)
        np.testing.assert_array_equal(model_samples[transition][:, 2:], targets[transition][:, 2:5])
        assert np.all(model_samples[~outer][:, 2:] == targets[~outer][:, 2:5])
        assert np.any(model_samples != samples)

    def test_shuffle_keeps_model_samples_aligned(self):
        params = {'num_base_sequences': 3, 'seed': 8, 'control_end': [True, False, True]}
        task, samples, model_samples, targets, df, _ = _generate(params, {'sequence_length': 120}, shuffle=True)
        np.testing.assert_array_equal(model_samples, task.model_samples)
        np.testing.assert_array_equal(samples[:, :, :2], targets[:, :, :2])
        # Dataframe rows still describe their arrays
        for i, row in df.iterrows():
            final = targets[i, -1, 2:5]
            assert {'red': 0, 'green': 1, 'blue': 2}[row['Final Color']] == int(np.argmax(final))


@pytest.mark.color_controlled_dataset
class TestOnDiskProduct:
    """P5/E: the save path (save_video_dataset) actually runs end to end."""

    def test_on_disk_layout(self, color_controlled_dataset_with_videos):
        d = color_controlled_dataset_with_videos
        dataset_dir = d['dataset_dir']
        df = d['df_data']

        assert (dataset_dir / 'trial_meta.csv').exists()
        assert (dataset_dir / 'dataset_meta.pkl').exists()
        # Blocks are mandatory: the LSTM loader cannot read save_video_dataset's
        # block-less fallback (HDS:922-925)
        blocks = sorted((dataset_dir / 'videos').glob('block_*'))
        assert [b.name for b in blocks] == ['block_1']

        assert set(df['Dataset Block']) == {1}
        assert sorted(df['Dataset Block Video']) == list(range(1, len(df) + 1))

        for _, row in df.iterrows():
            vdir = dataset_dir / 'videos' / f"block_{int(row['Dataset Block'])}" \
                / f"video_{int(row['Dataset Block Video'])}"
            for kind in ('samples', 'parameters', 'color_change'):
                path = vdir / f"{vdir.name}_{kind}.csv"
                assert path.exists(), path
                assert len(pd.read_csv(path)) == int(row['length'])
            assert (vdir / f"{vdir.name}_{row['Final Color']}.mp4").exists()

    def test_samples_csv_holds_model_samples(self, color_controlled_dataset_with_videos):
        """save_video_dataset writes df_model_sample to the *_samples.csv path
        (HDS:953), so the file the LSTM consumes as "samples" is model_samples."""
        d = color_controlled_dataset_with_videos
        df, model_samples = d['df_data'], d['model_samples']
        row = df.iloc[0]
        vdir = d['dataset_dir'] / 'videos' / f"block_{int(row['Dataset Block'])}" \
            / f"video_{int(row['Dataset Block Video'])}"
        on_disk = pd.read_csv(vdir / f"{vdir.name}_samples.csv", index_col=0)
        np.testing.assert_allclose(on_disk.values, model_samples[row['idx_trial']])

    def test_shuffled_rows_index_their_own_files(self, tmp_path):
        """With shuffle=True, `idx_trial` must index the in-memory arrays that
        match each row's own video CSVs (U4 SHOULD-FIX 3, verified on disk)."""
        from bouncing_ball_task.color_controlled_bouncing_ball.dataset import (
            generate_color_controlled_dataset_with_videos,
        )
        params = {'num_base_sequences': 2, 'seed': 77, 'duration': 50,
                  'variable_length': False, 'control_end': [False, True]}
        task_params = {'sequence_length': 60, 'size_frame': (256, 256), 'ball_radius': 10}
        funcs = {'no_change': generate_no_change_trials}

        task, samples, model_samples, targets, df, meta = \
            generate_color_controlled_dataset_with_videos(
                dict(params), dict(task_params), output_dir=tmp_path,
                shuffle=True, dict_trial_type_generation_funcs=funcs)
        dataset_dir = next(iter(tmp_path.glob('color_controlled_*')))

        # Premise: the shuffle actually reordered the trials
        _, _, _, _, df_unshuffled, _ = _generate(dict(params), dict(task_params))
        assert (df[['base_sequence_idx', 'color_rotation_idx']].values
                != df_unshuffled[['base_sequence_idx', 'color_rotation_idx']].values).any(), \
            "test premise: shuffle=True must reorder the trials"
        assert df['control_end'].any() and not df['control_end'].all(), \
            "test premise: both controlled and static trials on disk"

        for _, row in df.iterrows():
            idx = row['idx_trial']
            vdir = dataset_dir / 'videos' / f"block_{int(row['Dataset Block'])}" \
                / f"video_{int(row['Dataset Block Video'])}"
            parameters = pd.read_csv(vdir / f"{vdir.name}_parameters.csv", index_col=0)
            on_disk_samples = pd.read_csv(vdir / f"{vdir.name}_samples.csv", index_col=0)
            # The row's Final X/Y Position is the last row of ITS OWN CSV...
            np.testing.assert_allclose(
                parameters.values[-1, :2],
                [row['Final X Position'], row['Final Y Position']], atol=1e-3)
            # ...and idx_trial indexes the arrays that produced that file
            np.testing.assert_allclose(parameters.values, targets[idx], atol=1e-3)
            np.testing.assert_allclose(on_disk_samples.values, model_samples[idx], atol=1e-3)
            if row['control_end']:
                # control_end/base_sequence_idx must be permuted in step with the arrays
                requested = np.asarray(meta['controlled_parameters']['initial_position'])
                np.testing.assert_allclose(parameters.values[-1, :2],
                                           requested[int(row['base_sequence_idx'])], atol=1e-3)

    def test_variable_length_on_disk(self, tmp_path):
        """Variable-length datasets on disk: the CSVs hold exactly the LAST
        `length` frames of the full trajectory, so a reversed trial still ends at
        its requested position."""
        from bouncing_ball_task.color_controlled_bouncing_ball.dataset import (
            generate_color_controlled_dataset_with_videos,
        )
        params = {'num_base_sequences': 2, 'seed': 77, 'duration': 50,
                  'variable_length': True, 'control_end': [True, True],
                  # small human-sampler knobs (min 20 frames) to keep the videos short
                  'video_length_min_s': 1.0, 'exp_scale': 0.5}
        task_params = {'size_frame': (256, 256), 'ball_radius': 10}
        task, samples, model_samples, targets, df, meta = \
            generate_color_controlled_dataset_with_videos(
                dict(params), dict(task_params), output_dir=tmp_path, shuffle=True,
                dict_trial_type_generation_funcs={'no_change': generate_no_change_trials})
        dataset_dir = next(iter(tmp_path.glob('color_controlled_*')))

        assert df['length'].nunique() > 1, "test premise: lengths must vary across trials"
        duration = params['duration']
        min_length = int(meta['video_length_min_f'])
        requested = np.asarray(meta['controlled_parameters']['initial_position'])

        for _, row in df.iterrows():
            idx, length = row['idx_trial'], int(row['length'])
            vdir = dataset_dir / 'videos' / f"block_{int(row['Dataset Block'])}" \
                / f"video_{int(row['Dataset Block Video'])}"
            parameters = pd.read_csv(vdir / f"{vdir.name}_parameters.csv", index_col=0)
            assert len(parameters) == length, (idx, len(parameters), length)
            # The kept frames are the LAST `length` of the untruncated trajectory
            np.testing.assert_allclose(parameters.values, task.targets[idx, -length:], atol=1e-3)
            np.testing.assert_allclose(
                pd.read_csv(vdir / f"{vdir.name}_samples.csv", index_col=0).values,
                task.model_samples[idx, -length:], atol=1e-3)
            np.testing.assert_array_equal(parameters.index.values,
                                          np.arange(length) * duration)
            # ...so the controlled endpoint survives truncation and saving
            np.testing.assert_allclose(parameters.values[-1, :2],
                                       requested[int(row['base_sequence_idx'])], atol=1e-3)
            assert int(row['length_ms']) == length * duration
            # The window bound carries the length - min_length offset (HDS:376)
            assert int(row['last_visible_color_idx']) == _recompute_last_visible_color_idx(
                task.targets[idx, -length:, :2], length, min_length, task)

    def test_change_counts_match_the_targets_they_describe(self, multiple_variants_dataset):
        """The count columns come from compute_change_stats, which counts over
        target[:last_visible_color_idx] (HDS:546-549) — NOT the whole trajectory."""
        df = multiple_variants_dataset['df_data']
        targets = multiple_variants_dataset['targets']
        for i, row in df.iterrows():
            window = targets[row['idx_trial'], :int(row['last_visible_color_idx'])]
            assert row['Bounces'] == int(window[:, 5].sum()), i
            assert row['Random Bounces'] == int(window[:, 6].sum()), i
            assert row['Color Change Bounce'] == int(window[:, 7].sum()), i
            assert row['Color Change Random'] == int(window[:, 8].sum()), i
        assert df['Color Change Bounce'].sum() > 0, "test premise: some authored events"

    def test_last_visible_color_idx_recomputed_from_positions(self, multiple_variants_dataset):
        """`last_visible_color_idx` is the generator's own window bound, so the
        count columns must also be checked against a bound derived independently
        from positions + the outer mask geometry (taskutils.py:74-78)."""
        d = multiple_variants_dataset
        df, targets, task = d['df_data'], d['targets'], d['task']
        min_length = int(d['metadata']['video_length_min_f'])
        for i, row in df.iterrows():
            idx = row['idx_trial']
            expected_idx = _recompute_last_visible_color_idx(
                targets[idx, :, :2], row['length'], min_length, task)
            assert int(row['last_visible_color_idx']) == expected_idx, i
            # The recorded colour is the colour AT that frame
            assert int(row['color_entered']) == 1 + int(np.argmax(targets[idx, expected_idx, 2:5])), i
            # ...and the change counts are sums over that independent window
            window = targets[idx, :expected_idx]
            assert row['Bounces'] == int(window[:, 5].sum()), i
            assert row['Random Bounces'] == int(window[:, 6].sum()), i
            assert row['Color Change Bounce'] == int(window[:, 7].sum()), i
            assert row['Color Change Random'] == int(window[:, 8].sum()), i

    def test_trial_label_tracks_bounce_content(self, multiple_variants_dataset):
        """`trial` is derived from the trajectory, capitalised as on disk."""
        df = multiple_variants_dataset['df_data']
        targets = multiple_variants_dataset['targets']
        assert set(df['trial']) <= {'Bounce', 'Straight'}
        for i, row in df.iterrows():
            expected = 'Bounce' if targets[row['idx_trial'], :, 5].any() else 'Straight'
            assert row['trial'] == expected, i


@pytest.mark.color_controlled_dataset
def test_degenerate_last_visible_color_is_rejected():
    """A start just outside the outer grayzone band ([86 - 10, 170 + 10]) drifting
    inward slowly enough never to leave it again is visible only at frame 0, which
    would make the effective change statistics a 0/0 division inside
    compute_change_stats (HDS:628,638,643) — NaNs, not an error. Reachable only
    with an EXPLICIT start; the sampler never draws inside or near the grayzone."""
    with pytest.raises(ValueError, match="last visible at frame"):
        _generate({'num_base_sequences': 1, 'seed': 1,
                   'initial_position': [[74.0, 128.0]],
                   'initial_velocity': [[20.0, 0.0]]},
                  {'sequence_length': 30})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
