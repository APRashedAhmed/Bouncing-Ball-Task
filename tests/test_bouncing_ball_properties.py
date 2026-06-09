"""Regression tests for two latent BouncingBallTask defects (audit P4-2, P4-3)."""
import pytest

from bouncing_ball_task.bouncing_ball import BouncingBallTask


def test_str_does_not_raise():
    # P4-2: __str__ formatted self.min_t_color_change, an attribute that is never
    # set (it was split into the _after_random / _after_bounce pair), so any
    # str()/print()/log of a task raised AttributeError.
    task = BouncingBallTask(sequence_mode="reset")
    text = str(task)
    assert text.startswith("BouncingBallTask(")
    assert "min_t_color_change_after_random=" in text
    assert "min_t_color_change_after_bounce=" in text


@pytest.mark.parametrize("mode", ["half", "all"])
def test_transitioning_change_mode_reads_back(mode):
    # P4-3: the setter was registered under @return_change_mode.setter, so the
    # transitioning_change_mode property's getter aliased _return_change_mode and
    # reads returned the wrong value ('any' regardless of the constructed mode).
    task = BouncingBallTask(sequence_mode="reset", transitioning_change_mode=mode)
    assert task.transitioning_change_mode == mode


def test_transitioning_and_return_change_mode_are_independent():
    # Locks the aliasing bug: the two properties must not share a backing field.
    task = BouncingBallTask(
        sequence_mode="reset",
        transitioning_change_mode="half",
        return_change_mode="any",
    )
    assert task.transitioning_change_mode == "half"
    assert task.return_change_mode == "any"
