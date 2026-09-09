"""Tests for change vector generation and color-only authoring.

These tests validate the derived velocity ledger, the color-only authoring
contract and its contingency invariant, and the seeding contract of the
color-controlled dataset.
"""
import pytest
import numpy as np

from bouncing_ball_task.color_controlled_bouncing_ball import dataset as cds
from bouncing_ball_task.color_controlled_bouncing_ball.dataset import generate_color_controlled_dataset
from bouncing_ball_task.color_controlled_bouncing_ball.no_change import generate_no_change_trials


def _generate(color_controlled_params, task_params, variants=None):
    return generate_color_controlled_dataset(
        color_controlled_params,
        task_params,
        shuffle=False,
        dict_trial_type_generation_funcs=variants or {'no_change': generate_no_change_trials},
    )


def positions_derived_bounce_mask(positions):
    """TEST-ONLY cross-check: a bounce is a sign flip of the position diff.

    Returns (N, T) bool with the flag at turning point + 1, the base task's
    convention (BB:1470-1621). Never the runtime invariant (forced bounces
    would be missed).
    """
    diff = np.diff(positions, axis=1)  # (N, T-1, 2)
    flip = np.sign(diff[:, 1:]) != np.sign(diff[:, :-1])  # (N, T-2, 2)
    mask = np.zeros(positions.shape[:2], dtype=bool)
    mask[:, 2:] = np.any(flip, axis=-1)
    return mask


@pytest.mark.color_controlled_dataset
class TestChangeVectorGeneration:
    """Test change vector generation and manipulation."""
    
    def test_change_vector_shape(self, color_controlled_dataset_small):
        """Verify change vectors have correct shape."""
        targets = color_controlled_dataset_small['targets']
        
        # Change vectors should be in indices 5:9 of targets
        # Shape should be (N, T, 4) where:
        # - N is number of trials
        # - T is sequence length
        # - 4 is [bounce_velocity, random_velocity, color_bounce, color_random]
        
        for i, target in enumerate(targets):
            change_vector = target[:, 5:9]
            assert change_vector.shape[1] == 4, \
                f"Change vector should have 4 channels, got {change_vector.shape[1]}"
    
    def test_base_sequence_reuse(self, color_controlled_dataset_small):
        """Verify that base sequences are reused across color rotations."""
        df_data = color_controlled_dataset_small['df_data']
        targets = color_controlled_dataset_small['targets']
        
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
    
    def test_color_change_zeroing(self, color_controlled_dataset_small):
        """Verify the no_change variant authors zero color events (channels 7, 8)."""
        targets = color_controlled_dataset_small['targets']
        
        # Color change indicators are at indices 7 and 8
        for i, target in enumerate(targets):
            color_changes = target[:, 7:9]
            
            assert np.all(color_changes == 0), \
                f"Color changes should be zeroed in trial {i}"

    @pytest.fixture
    def mixed_mode_dataset(self):
        """Static AND reversed trials: reverse frames 0/1 come from the
        reverse_ball_sequence shift (BB:1740) and need their own coverage."""
        return _generate({'num_base_sequences': 4, 'seed': 7,
                          'control_end': [False, True, False, True]},
                         {'sequence_length': 120})

    def test_frame_zero_is_not_a_changepoint(self, mixed_mode_dataset):
        """Frame 0 of the change vector is all zeros — parameter parity
        (initial_timestep_is_changepoint=False), never a runtime transform."""
        _, _, _, targets, _, metadata = mixed_mode_dataset
        assert metadata['task_parameters']['initial_timestep_is_changepoint'] is False
        assert np.all(targets[:, 0, 5:9] == 0), "frame 0 must carry no change flags"
        # Reverse-mode trials additionally carry no flag at frame 1 (the +2 shift)
        reverse_rows = metadata['controlled_parameters']['control_end']
        assert reverse_rows.any(), "test premise: some trials must be reversed"

    def test_random_velocity_ledger_is_zero(self, mixed_mode_dataset):
        """ch6 (random velocity change) is identically zero under PVC=0."""
        _, _, _, targets, _, metadata = mixed_mode_dataset
        assert metadata['task_parameters']['probability_velocity_change'] == 0.0
        assert np.all(targets[:, :, 6] == 0)

    def test_nonzero_pvc_is_rejected(self):
        """PVC != 0 breaks 'trajectory is a function of initial conditions'."""
        with pytest.raises(ValueError, match="probability_velocity_change"):
            _generate({'num_base_sequences': 2, 'seed': 1},
                      {'sequence_length': 30, 'probability_velocity_change': 0.075})

    def test_initial_timestep_is_changepoint_true_is_rejected(self):
        """A True frame-0 changepoint flags a bounce at frame 0 for every trial,
        which would let a variant author a contingent color change there and
        break ch6 == 0. Rejected by symmetry with the PVC guard."""
        with pytest.raises(ValueError, match="initial_timestep_is_changepoint"):
            _generate({'num_base_sequences': 2, 'seed': 1},
                      {'sequence_length': 30, 'initial_timestep_is_changepoint': True})
    
    def test_cumulative_color_generation(self, color_controlled_dataset_small):
        """Test the cumulative sum approach for color generation."""
        targets = color_controlled_dataset_small['targets']
        samples = color_controlled_dataset_small['samples']
        
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
    
    def test_position_extraction(self, color_controlled_dataset_small):
        """Verify that positions are correctly extracted from targets."""
        targets = color_controlled_dataset_small['targets']
        samples = color_controlled_dataset_small['samples']
        
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
        num_base = multiple_variants_dataset['color_controlled_params']['num_base_sequences']
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
    
    def test_color_rotation_pattern(self, color_controlled_dataset_small):
        """Verify the pattern of color rotations."""
        df_data = color_controlled_dataset_small['df_data']
        
        # Group by color rotation
        for color_idx in [0, 1, 2]:
            color_trials = df_data[df_data['color_rotation_idx'] == color_idx]
            
            # Each color rotation should have all base sequences
            base_sequences = sorted(color_trials['base_sequence_idx'].unique())
            expected_bases = list(range(color_controlled_dataset_small['color_controlled_params']['num_base_sequences']))
            
            assert base_sequences == expected_bases, \
                f"Color rotation {color_idx} missing base sequences"
    
    def test_final_target_shape(self, color_controlled_dataset_small):
        """Verify final target array has correct shape."""
        targets = color_controlled_dataset_small['targets']
        df_data = color_controlled_dataset_small['df_data']
        
        num_trials = len(df_data)
        sequence_length = color_controlled_dataset_small['task_params']['sequence_length']
        
        expected_shape = (num_trials, sequence_length, 9)
        assert targets.shape == expected_shape, \
            f"Targets shape {targets.shape} doesn't match expected {expected_shape}"
        
        # Verify the 9 channels are:
        # [x, y, r, g, b, bounce_vc, random_vc, color_bounce, color_random]
        assert targets.shape[2] == 9, "Targets should have 9 channels"
    
    def test_base_task_properties(self, color_controlled_dataset_small):
        """Verify base task properties are preserved."""
        task = color_controlled_dataset_small['task']
        task_params = color_controlled_dataset_small['task_params']
        
        # Check task has expected properties
        assert task.sequence_length == task_params['sequence_length']
        assert np.array_equal(task.size_frame, task_params['size_frame'])
        assert task.ball_radius == task_params['ball_radius']
        
        # Check samples and targets match
        assert len(task.samples) == len(task.targets)
        assert task.samples.shape[1] == task.sequence_length
        assert task.targets.shape[1] == task.sequence_length


@pytest.mark.color_controlled_dataset
class TestBaseTaskNotMutated:
    """Regression test: generate_color_controlled_dataset must not mutate base task targets.

    The base task's naturally-generated change record (its ``targets``) is the
    read-only ledger. Extracting the velocity channels as a view and writing
    into the extracted array would silently corrupt it; the generator must
    copy at extraction.
    """

    def test_base_task_targets_color_channels_not_zeroed(self, monkeypatch):
        """Every base BouncingBallTask's targets survive dataset generation.

        Capture each ``BouncingBallTask`` built inside the generator together
        with a snapshot of its ``targets`` at construction; after generation
        the live ``targets`` must equal the snapshot. The base task is driven
        with a high natural color-change probability so its channels 7:9 are
        non-zero — i.e. would visibly change if the generator wrote into them.
        """
        import bouncing_ball_task.color_controlled_bouncing_ball.dataset as dataset_module

        captured = []
        real_bouncing_ball_task = dataset_module.BouncingBallTask

        def capturing_bouncing_ball_task(*args, **kwargs):
            instance = real_bouncing_ball_task(*args, **kwargs)
            if kwargs.get('sequence_mode') in ('static', 'reverse') and kwargs.get('sequence_length', 0) > 2:
                captured.append((instance, instance.targets.copy()))
            return instance

        monkeypatch.setattr(dataset_module, 'BouncingBallTask', capturing_bouncing_ball_task)

        color_controlled_params = {
            'num_base_sequences': 5,
            'seed': 42,
            'duration': 1000,
            'variable_length': False,
            'control_end': [False, True, False, True, False],
        }

        task_params = {
            'sequence_length': 100,
            'size_frame': (256, 256),
            'ball_radius': 10,
            # Force plenty of natural color changes on the base task so its
            # targets[:, :, 7:9] (color-change channels) are non-zero.
            'probability_color_change_no_velocity_change': 0.5,
            'probability_color_change_on_velocity_change': 0.5,
        }

        _, _, _, final_targets, _, _ = _generate(color_controlled_params, task_params)

        assert len(captured) == 2, "expected one forward and one reverse base task"
        assert any(np.any(snapshot[:, :, 7:9] != 0) for _, snapshot in captured), \
            "test premise: base tasks must produce natural color changes"
        for base_task, snapshot in captured:
            np.testing.assert_array_equal(
                base_task.targets, snapshot,
                err_msg="base task targets were mutated by generate_color_controlled_dataset",
            )
        # ...while the final targets carry the authored (all-zero) color channels
        assert np.all(final_targets[:, :, 7:9] == 0)


@pytest.mark.color_controlled_dataset
class TestContingencyInvariant:
    """P4: contingent color changes fire only on real bounces; never both channels."""

    @pytest.fixture
    def ledger(self):
        params = {'num_base_sequences': 4, 'seed': 7, 'control_end': [False, True, False, True]}
        _, _, _, targets, df, _ = _generate(params, {'sequence_length': 120})
        first = df.drop_duplicates('base_sequence_idx').sort_values('base_sequence_idx')
        base = targets[first.index.values]
        return base[:, :, :2], base[:, :, 5].astype(bool)

    def test_saved_ledger_satisfies_invariant(self, multiple_variants_dataset):
        """On the saved targets: ch7 => ch5 and ch7 + ch8 <= 1, every trial/frame."""
        targets = multiple_variants_dataset['targets']
        ch5, ch7, ch8 = targets[:, :, 5], targets[:, :, 7], targets[:, :, 8]
        assert np.all(ch5[ch7 == 1] == 1), "contingent color change without a bounce"
        assert np.all(ch7 + ch8 <= 1), "both color channels fired on one frame"
        # The test variant fires on every bounce, so the check is not vacuous
        test_rows = (multiple_variants_dataset['df_data']['variant'] == 'test_variant').values
        assert ch7[test_rows].sum() > 0 and np.array_equal(ch7[test_rows], ch5[test_rows])

    def test_bounce_ledger_matches_positions_cross_check(self, ledger):
        """TEST-ONLY cross-check: ch5 == sign flips of the position diff."""
        positions, bounce_frames = ledger
        assert bounce_frames.sum() > 0
        np.testing.assert_array_equal(bounce_frames, positions_derived_bounce_mask(positions))

    def test_orphan_contingent_change_raises(self, ledger):
        positions, bounce_frames = ledger
        events = np.zeros(bounce_frames.shape + (2,), dtype=int)
        trial, frame = np.argwhere(~bounce_frames)[10]
        events[trial, frame, 0] = 1
        with pytest.raises(ValueError, match="no bounce occurred"):
            cds.validate_color_events(events, bounce_frames, 'bad')

    def test_double_fire_raises(self, ledger):
        positions, bounce_frames = ledger
        events = np.zeros(bounce_frames.shape + (2,), dtype=int)
        trial, frame = np.argwhere(bounce_frames)[0]
        events[trial, frame, :] = 1
        with pytest.raises(ValueError, match="both contingent and random"):
            cds.validate_color_events(events, bounce_frames, 'bad')

    def test_wrong_shape_or_nonbinary_raises(self, ledger):
        positions, bounce_frames = ledger
        with pytest.raises(ValueError, match="shape"):
            cds.validate_color_events(np.zeros(bounce_frames.shape + (4,)), bounce_frames, 'bad')
        with pytest.raises(ValueError, match="binary"):
            cds.validate_color_events(np.full(bounce_frames.shape + (2,), 0.5), bounce_frames, 'bad')

    def test_violating_variant_raises_end_to_end(self):
        def bad_variant(positions, bounce_frames, params):
            events = np.zeros(bounce_frames.shape + (2,), dtype=int)
            events[:, :, 0] = ~bounce_frames  # contingent where there is NO bounce
            return events

        with pytest.raises(ValueError, match="no bounce occurred"):
            _generate({'num_base_sequences': 2, 'seed': 3}, {'sequence_length': 40},
                      variants={'bad': bad_variant})

    def test_random_channel_is_free_of_the_bounce_rule(self):
        def random_variant(positions, bounce_frames, params):
            events = np.zeros(bounce_frames.shape + (2,), dtype=int)
            events[:, 10, 1] = 1
            return events

        _, _, _, targets, _, _ = _generate({'num_base_sequences': 2, 'seed': 3},
                                           {'sequence_length': 40},
                                           variants={'rnd': random_variant})
        assert np.all(targets[:, 10, 8] == 1)
        # The authored event drives the color sequence (cumsum + rotation)
        assert not np.array_equal(targets[0, 9, 2:5], targets[0, 10, 2:5])


@pytest.mark.color_controlled_dataset
class TestSeedingContract:
    """P3: seed once -> resolve init conditions -> both sub-batches with seed=False."""

    PARAMS = {'num_base_sequences': 6, 'seed': 2024,
              'control_end': [False, True, False, True, False, True]}
    TASK = {'sequence_length': 60}

    def test_same_seed_is_byte_identical(self):
        a = _generate(dict(self.PARAMS), dict(self.TASK))
        b = _generate(dict(self.PARAMS), dict(self.TASK))
        for idx in (1, 2, 3):  # samples, model_samples, targets
            assert a[idx].tobytes() == b[idx].tobytes()
        np.testing.assert_array_equal(
            a[5]['controlled_parameters']['initial_position'],
            b[5]['controlled_parameters']['initial_position'])

    def test_forward_and_reverse_subbatches_have_distinct_starts(self):
        """Guards the seeding fix: reseeding per sub-batch duplicated sampled starts."""
        _, _, _, targets, df, meta = _generate(dict(self.PARAMS), dict(self.TASK))
        starts = meta['controlled_parameters']['initial_position']
        control_end = meta['controlled_parameters']['control_end']
        fwd, rev = starts[~control_end], starts[control_end]
        assert len(np.unique(starts.round(6), axis=0)) == len(starts)
        assert not np.any(np.all(np.isclose(fwd[:, None, :], rev[None, :, :]), axis=-1))
        # ...and the trajectories reflect it: static starts vs reversed ends
        first = df.drop_duplicates('base_sequence_idx').sort_values('base_sequence_idx')
        base = targets[first.index.values]
        controlled = np.where(control_end[:, None], base[:, -1, :2], base[:, 0, :2])
        np.testing.assert_allclose(controlled, starts, atol=1e-3)

    def test_sub_batch_tasks_never_reseed(self, monkeypatch):
        """Every BouncingBallTask gets seed=False AND the global RNG is seeded
        exactly once (the dataset anchor). The seed=False check alone would pass
        even if the generator called set_global_seed itself mid-pipeline — the
        exact trap BB:237-239 sets — so the call count is asserted too."""
        import bouncing_ball_task.color_controlled_bouncing_ball.dataset as dataset_module

        seeds = []
        real = dataset_module.BouncingBallTask

        def capturing(*args, **kwargs):
            seeds.append(kwargs.get('seed', 'MISSING'))
            return real(*args, **kwargs)

        seedings = []
        real_set_global_seed = dataset_module.pyutils.set_global_seed

        def counting_set_global_seed(value=None):
            seedings.append(value)
            return real_set_global_seed(value)

        monkeypatch.setattr(dataset_module, 'BouncingBallTask', capturing)
        monkeypatch.setattr(dataset_module.pyutils, 'set_global_seed',
                            counting_set_global_seed)
        _generate(dict(self.PARAMS), dict(self.TASK))
        assert seeds and all(seed is False for seed in seeds), seeds
        assert seedings == [self.PARAMS['seed']], seedings

    def test_seed_false_escape_hatch_defers_to_caller(self):
        np.random.seed(11)
        a = _generate({**self.PARAMS, 'seed': False}, dict(self.TASK))
        np.random.seed(11)
        b = _generate({**self.PARAMS, 'seed': False}, dict(self.TASK))
        assert a[3].tobytes() == b[3].tobytes()
        assert a[5]['resolved_seed'] is None


@pytest.mark.color_controlled_dataset
class TestInitialConditionValidation:

    def test_out_of_bounds_start_raises(self):
        params = {'num_base_sequences': 2, 'seed': 1,
                  'initial_position': [[5.0, 100.0], [100.0, 100.0]]}  # x=5 < r=10
        with pytest.raises(ValueError, match="strictly inside"):
            _generate(params, {'sequence_length': 30})

    def test_boundary_start_raises(self):
        params = {'num_base_sequences': 1, 'seed': 1, 'initial_position': [[10.0, 246.0]]}
        with pytest.raises(ValueError, match="strictly inside"):
            _generate(params, {'sequence_length': 30, 'ball_radius': 10, 'size_frame': (256, 256)})

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _generate({'num_base_sequences': 3, 'seed': 1, 'initial_position': [[50.0, 50.0]]},
                      {'sequence_length': 30})
        with pytest.raises(ValueError, match="control_end"):
            _generate({'num_base_sequences': 3, 'seed': 1, 'control_end': [True]},
                      {'sequence_length': 30})

    def test_explicit_velocity_with_sampled_position(self):
        params = {'num_base_sequences': 2, 'seed': 5,
                  'initial_velocity': [[20.0, 15.0], [-20.0, -15.0]]}
        _, _, _, targets, _, meta = _generate(params, {'sequence_length': 30})
        np.testing.assert_array_equal(meta['controlled_parameters']['initial_velocity'],
                                      params['initial_velocity'])
        # First step follows the explicit velocity (dt inherited from the real task)
        dt = meta['task_parameters']['dt']
        step = targets[[0, 1], 1, :2] - targets[[0, 1], 0, :2]
        np.testing.assert_allclose(step, np.asarray(params['initial_velocity']) * dt, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
