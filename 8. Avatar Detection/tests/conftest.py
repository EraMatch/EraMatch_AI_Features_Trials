"""Shared pytest fixtures for avatar detection module tests."""

import tempfile
from pathlib import Path

import pytest
import torch


@pytest.fixture
def tmp_results_dir():
    """Create and clean up a temporary directory for test results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image_tensor():
    """Return a single random image tensor (1, 3, 256, 256)."""
    return torch.randn(1, 3, 256, 256)


@pytest.fixture
def sample_batch():
    """Return a sample batch dict mimicking DataLoader output.

    Returns:
        dict with keys:
            image: (B, 3, 256, 256) float tensor
            label: (B,) long tensor (0=real, 1=fake)
            dataset: (B,) long tensor (dataset source id)
            path: (B,) list of string paths
    """
    batch_size = 4
    return {
        "image": torch.randn(batch_size, 3, 256, 256),
        "label": torch.tensor([0, 1, 0, 1]),
        "dataset": torch.tensor([0, 0, 1, 1]),
        "path": [f"sample_{i}.jpg" for i in range(batch_size)],
    }
