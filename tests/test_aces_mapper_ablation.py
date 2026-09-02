"""Tests for the ACESMapper inference-time refiner ablation (use_refiner=False).

Ablation A of the MED8 artifact diagnosis: the Laplacian refinement head must
be skippable at inference without affecting checkpoint compatibility.
"""

from __future__ import annotations

import pytest
import torch

from luminascale.models.aces_mapper import ACESMapper

SMALL_KWARGS = dict(
    num_luts=2,
    lut_dim=17,
    num_lap=2,
    num_residual_blocks=1,
    sft_embed_dim=16,  # must stay divisible by the largest head count
    sft_num_heads=[1, 2, 4, 8, 8, 4, 2, 1],
)


@pytest.fixture(scope="module")
def seed() -> None:
    torch.manual_seed(9)


def _make_mapper(use_refiner: bool) -> ACESMapper:
    torch.manual_seed(9)  # identical init for both modes
    return ACESMapper(use_refiner=use_refiner, **SMALL_KWARGS)


def test_lut_only_output_shape() -> None:
    model = _make_mapper(use_refiner=False)
    model.eval()
    dummy = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        out, point_weights = model(dummy)
    assert out.shape == dummy.shape
    assert point_weights.shape == (1, SMALL_KWARGS["num_luts"])


def test_lut_only_differs_from_refined_path() -> None:
    refined = _make_mapper(use_refiner=True)
    lut_only = _make_mapper(use_refiner=False)
    refined.eval()
    lut_only.eval()
    dummy = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        out_refined, _ = refined(dummy)
        out_lut_only, _ = lut_only(dummy)
    # Identical init => identical LUT stage; the refiner must change the output.
    assert not torch.allclose(out_refined, out_lut_only, atol=1e-5), (
        "Refiner had no effect on the output; ablation would be a no-op"
    )


def test_refiner_flag_keeps_state_dict_compatible() -> None:
    refined = _make_mapper(use_refiner=True)
    lut_only = _make_mapper(use_refiner=False)
    # A checkpoint trained with the refiner must load into a LUT-only model
    # with strict=True and vice versa (flag is forward-only, not structural).
    lut_only.load_state_dict(refined.state_dict(), strict=True)
    refined.load_state_dict(lut_only.state_dict(), strict=True)


def test_lut_only_is_deterministic(seed: None) -> None:
    model = _make_mapper(use_refiner=False)
    model.eval()
    dummy = torch.rand(1, 3, 128, 128)
    with torch.inference_mode():
        first, _ = model(dummy)
        second, _ = model(dummy)
    assert torch.allclose(first, second, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
