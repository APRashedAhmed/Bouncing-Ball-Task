"""Schema-parity acceptance spec for the color-controlled dataset generator.

The generator must write datasets in the on-disk format of the real human
bouncing-ball task. The reference schema is a snapshot of
``/mnt/work/data/raw/hbb_v3_2_2/hbb_dataset_v3_2_2`` (what the LSTM actually
consumed), stored in ``tests/fixtures/v3_2_2_schema.json``.

Parity definition (pre-decided; do not weaken):
  (a) (real columns - omitted_columns) is a subset of generated columns
  (b) dtypes are equal on the intersection
  (c) extra generated columns are tolerated
  (d) trial_meta index is named ``Video ID``
  (e) per-video CSV headers equal the snapshot
  (f) dataset_meta.pkl["task_parameters"] keys are a superset of the snapshot's
  (g) per-video CSV row count == that trial's ``length``

The tests were authored ``xfail(strict=True)`` as the P5 forcing function; the
markers were removed once the generator met the spec.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "v3_2_2_schema.json"

pytestmark = pytest.mark.color_controlled_dataset


@pytest.fixture(scope="module")
def schema():
    with open(FIXTURE_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def generated_dataset(tmp_path_factory):
    """Generate a small dataset with videos into a temp dir and locate its files.

    Imports happen inside the fixture: the package is mid-rewrite and a
    top-level ImportError would be a collection error, not an xfail.
    """
    from bouncing_ball_task.color_controlled_bouncing_ball.dataset import (
        generate_color_controlled_dataset_with_videos,
    )
    from bouncing_ball_task.color_controlled_bouncing_ball.no_change import (
        generate_no_change_trials,
    )

    output_dir = tmp_path_factory.mktemp("cc_parity")
    color_controlled_params = {
        "num_base_sequences": 3,
        "seed": 123,
        "duration": 50,
        "variable_length": False,
    }
    task_params = {
        "sequence_length": 50,
        "size_frame": (256, 256),
        "ball_radius": 10,
    }
    generate_color_controlled_dataset_with_videos(
        color_controlled_params,
        task_params,
        output_dir=output_dir,
        shuffle=False,
        dict_trial_type_generation_funcs={"no_change": generate_no_change_trials},
    )

    trial_metas = list(output_dir.rglob("trial_meta.csv"))
    assert len(trial_metas) == 1, f"expected one trial_meta.csv, found {trial_metas}"
    dataset_dir = trial_metas[0].parent
    assert (dataset_dir / "dataset_meta.pkl").exists(), "dataset_meta.pkl missing"
    assert (dataset_dir / "videos").is_dir(), "videos/ directory missing"

    df = pd.read_csv(dataset_dir / "trial_meta.csv", index_col=0)
    with open(dataset_dir / "dataset_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    return {"dataset_dir": dataset_dir, "df": df, "meta": meta}


def _required_columns(schema):
    omitted = set(schema["omitted_columns"])
    return [c for c in schema["trial_meta"]["columns"] if c not in omitted]


def test_trial_meta_columns_superset(schema, generated_dataset):
    """(a) + (c): required real columns present; extras tolerated."""
    df = generated_dataset["df"]
    missing = [c for c in _required_columns(schema) if c not in df.columns]
    assert not missing, f"trial_meta.csv is missing real-task columns: {missing}"


def test_trial_meta_dtypes_match_on_intersection(schema, generated_dataset):
    """(b): pandas dtype strings equal on every shared column (sentinel dtypes included)."""
    df = generated_dataset["df"]
    real_dtypes = schema["trial_meta"]["dtypes"]
    shared = [c for c in real_dtypes if c in df.columns]
    assert shared, "no shared columns between generated trial_meta and snapshot"
    mismatched = {
        c: (real_dtypes[c], str(df[c].dtype))
        for c in shared
        if str(df[c].dtype) != real_dtypes[c]
    }
    assert not mismatched, f"dtype mismatches (real, generated): {mismatched}"


def test_trial_meta_index_name(schema, generated_dataset):
    """(d): index named 'Video ID' when read with index_col=0."""
    assert generated_dataset["df"].index.name == schema["trial_meta"]["index_name"]


def test_video_csv_headers(schema, generated_dataset):
    """(e): each per-video CSV header equals the snapshot (Timestamp included)."""
    dataset_dir = generated_dataset["dataset_dir"]
    expected = schema["video_csvs"]["files"]
    video_dirs = sorted(p for p in dataset_dir.glob("videos/block_*/video_*") if p.is_dir())
    assert video_dirs, "no videos/block_N/video_M directories found"
    for vdir in video_dirs:
        for kind, spec in expected.items():
            csv_path = vdir / f"{vdir.name}_{kind}.csv"
            assert csv_path.exists(), f"missing {csv_path}"
            header = list(pd.read_csv(csv_path, nrows=0).columns)
            assert header == spec["columns"], f"{csv_path}: {header} != {spec['columns']}"


def test_dataset_meta_task_parameter_keys(schema, generated_dataset):
    """(f): task_parameters keys are a superset of the real snapshot's keys."""
    meta = generated_dataset["meta"]
    assert "task_parameters" in meta, f"dataset_meta.pkl keys: {list(meta)}"
    missing = set(schema["dataset_meta"]["task_parameters_keys"]) - set(meta["task_parameters"])
    assert not missing, f"task_parameters missing keys: {sorted(missing)}"


def test_length_equals_frame_count(generated_dataset):
    """(C2) `length` is a FRAME COUNT: for a fixed-length dataset it is exactly
    the task's sequence_length, and `length_ms = length * duration` with
    `duration` in ms PER FRAME."""
    df = generated_dataset["df"]
    meta = generated_dataset["meta"]
    sequence_length = meta["task_parameters"]["sequence_length"]
    assert (df["length"] == sequence_length).all(), df["length"].unique()
    duration = meta["duration"]
    assert (df["length_ms"] == df["length"] * duration).all()


def test_video_csv_row_count_equals_length(schema, generated_dataset):
    """(g): every per-video CSV has exactly `length` rows for its trial."""
    dataset_dir = generated_dataset["dataset_dir"]
    df = generated_dataset["df"]
    for col in ("Dataset Block", "Dataset Block Video", "length"):
        assert col in df.columns, f"trial_meta.csv lacks {col}"
    bad = []
    for _, row in df.iterrows():
        block = int(row["Dataset Block"])
        vid = int(row["Dataset Block Video"])
        vdir = dataset_dir / "videos" / f"block_{block}" / f"video_{vid}"
        for kind in schema["video_csvs"]["files"]:
            csv_path = vdir / f"video_{vid}_{kind}.csv"
            if not csv_path.exists():
                bad.append((str(csv_path), "missing"))
                continue
            n = len(pd.read_csv(csv_path))
            if n != int(row["length"]):
                bad.append((str(csv_path), f"{n} rows != length {int(row['length'])}"))
    assert not bad, f"row-count mismatches: {bad}"
