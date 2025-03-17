import pytest
from bouncing_ball_task.utils.pyutils import create_sequence_splits

# Each test case is a tuple: (input_sequence, total, expected_raw, expected_normalized)
@pytest.mark.parametrize("seq, total, expected_values", [
    ([1, 1, -2, -1, -1], 10, [0.1, 0.1, 0.4, 0.2, 0.2]),
    ([1, 1, None, -1, -1], 10, [0.1, 0.1, (8/3)/10, (8/3)/10, (8/3)/10]),
    ([0.3, 0.3, 0.2], None, [0.3/0.8, 0.3/0.8, 0.2/0.8]),
    ([-1, -2], None, [1/3, 2/3]),
    ([1, -2], None, [1.0, 0.0]),
    ([10, -1], 10, [1.0, 0.0]),
    ([0.05, -1, -1, -1], None,  [0.05,] + [(1 - 0.05)/3]*3),
    ([5, -1, -1, -1], 100,  [0.05,] + [(1 - 0.05)/3]*3),
    ([0.0, -1, -1, -1], None,  [0.0,] + [1/3]*3),
])
def test_create_sequence_splits_norm_false(seq, total, expected_values):
    """
    Test create_sequence_splits with norm set to False.
    This test uses both cases where total is specified and where it defaults to 1.0.
    """
    result = create_sequence_splits(seq, total=total)
    # Compare each element using pytest.approx for floating-point precision.
    for r, expected in zip(result, expected_values):
        assert pytest.approx(r) == expected

@pytest.mark.parametrize("seq, total", [
    # Fixed sum greater than total should raise an error.
    ([2, 2, -1], None),
    ([2, 2, -1], 3),
])
def test_create_sequence_splits_errors(seq, total):
    with pytest.raises(ValueError):
        create_sequence_splits(seq, total=total)        
