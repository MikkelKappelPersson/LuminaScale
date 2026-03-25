import torch

import numpy as np
from PIL import Image

from luminascale.utils.io import image_to_tensor


def test_image_to_tensor(tmp_path):
    # create a simple 2x2 image with known values
    arr = np.array(
        [[[0, 128, 255], [255, 128, 0]], [[64, 64, 64], [192, 192, 192]]],
        dtype=np.uint8,
    )
    img_path = tmp_path / "sample.png"
    Image.fromarray(arr).save(img_path)

    tensor = image_to_tensor(img_path)

    # shape should be [C, H, W]
    assert tuple(tensor.shape) == (3, 2, 2)
    assert tensor.dtype == torch.float32

    # values should be in [0,1] and approximately equal to arr/255
    expected = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    assert torch.allclose(tensor, expected, atol=1e-6)
