# Controlled Dataset Variant Tests

This directory contains tests for controlled dataset variants. The structure is designed to be scalable and maintainable as new variants are added.

## Structure

- `base_variant_tests.py` - Base test class containing tests that all variants should pass
- `test_no_change.py` - Tests specific to the no_change variant
- `conftest.py` - Fixtures specific to variant testing

## Running Tests

```bash
# Run all variant tests
pytest tests/color_controlled_variants/

# Run tests for a specific variant
pytest -m no_change

# Run all variant tests (current and future)
pytest -m variant

# Run specific test file
pytest tests/color_controlled_variants/test_no_change.py
```

## Adding New Variant Tests

When implementing a new variant (e.g., sudden_change), follow these steps:

1. Create `test_sudden_change.py` in this directory
2. Import and inherit from `BaseVariantTests`
3. Add the appropriate pytest markers (`@pytest.mark.variant`, `@pytest.mark.sudden_change`)
4. Override the `variant_dataset` fixture to provide your variant's dataset
5. Set the `variant_functions` class attribute (`{name: authoring_function}`) so the
   base contract test can call every variant in the dataset
6. Add variant-specific tests as needed

Variant authoring contract (color-only):
```python
def generate_<variant>_trials(positions, bounce_frames, params) -> np.ndarray  # (N, T, 2)
```
`positions` `(N, T, 2)` and `bounce_frames` `(N, T)` (targets channel 5) are read-only; the
return holds `[cc_bounce (ch 7), cc_random (ch 8)]`, binary, with `ch7 => ch5` and
`ch7 + ch8 <= 1` enforced by the generator. Velocity channels are never authorable.

Example structure:
```python
from .base_variant_tests import BaseVariantTests

@pytest.mark.color_controlled_dataset
@pytest.mark.variant
@pytest.mark.sudden_change
class TestSuddenChangeVariant(BaseVariantTests):
    variant_functions = {'sudden_change': generate_sudden_change_trials}

    @pytest.fixture
    def variant_dataset(self, sudden_change_dataset):
        return sudden_change_dataset
    
    # Add variant-specific tests here
```

## Common Tests (from BaseVariantTests)

All variants automatically get these tests:
- `test_position_consistency_across_color_rotations` - Ensures positions are identical across color rotations
- `test_metadata_structure` - Verifies proper metadata structure
- `test_three_color_rotation_pattern` - Checks for correct RGB rotations
- `test_variant_authors_color_channels_only` - Validates the color-only `(N, T, 2)` authoring contract
- `test_dataframe_completeness` - Ensures all required columns exist
- `test_trial_count_formula` - Verifies correct number of trials

## Benefits

1. **No code duplication** - Common tests are inherited
2. **Easy to add variants** - Just inherit and add specific tests
3. **Clear organization** - Each variant has its own test file
4. **Flexible execution** - Run by variant, all variants, or specific files
5. **Scalable** - Structure supports many variants without clutter
