"""Regression test for the save_gif VideoWriter resource leak (audit P5-6)."""
import numpy as np
import pytest

import bouncing_ball_task.utils.gif as gif_mod


class _FakeVideoWriter:
    """Stand-in whose write() always raises, recording whether release() ran."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.released = False
        _FakeVideoWriter.instances.append(self)

    def write(self, frame):
        raise RuntimeError("simulated codec failure")

    def release(self):
        self.released = True


def test_save_gif_releases_videowriter_on_write_error(tmp_path, monkeypatch):
    # P5-6: video.release() sat outside any try/finally, so an exception in the
    # write loop (bad frame shape, codec failure) leaked the handle and could
    # corrupt the output file.
    _FakeVideoWriter.instances = []
    monkeypatch.setattr(gif_mod.cv2, "VideoWriter", _FakeVideoWriter)

    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

    with pytest.raises(RuntimeError, match="simulated codec failure"):
        gif_mod.save_gif(
            images,
            path_dir=str(tmp_path),
            name="leak_test",
            duration=1000,
            include_timestep=False,
            as_mp4=True,
        )

    assert _FakeVideoWriter.instances, "VideoWriter was never constructed"
    assert _FakeVideoWriter.instances[0].released, (
        "VideoWriter.release() must run even when a frame write raises"
    )
