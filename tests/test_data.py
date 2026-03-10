from __future__ import annotations

import numpy as np

from mlfd.data import convert_field_layout


def test_convert_field_layout_uses_vector_reshape_not_transpose() -> None:
    field = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    converted = convert_field_layout(field, 3, 2)
    expected = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    assert np.array_equal(converted, expected)
