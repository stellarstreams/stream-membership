import numpy as np

from stream_membership.utils import get_coord_from_data_dict


def test_get_coord_from_data_dict():
    data = {"phi1": np.array(1), "phi2": np.array(2)}
    assert get_coord_from_data_dict("phi1", data) == 1

    data = {("phi1", "phi2"): np.array([1, 2])}
    assert get_coord_from_data_dict("phi1", data) == 1
    assert get_coord_from_data_dict("phi2", data) == 2
    assert get_coord_from_data_dict(("phi4", "phi2"), data) is None

    data = {("phi1", "phi2"): np.array([[1, 2], [3, 4]])}
    assert np.allclose(get_coord_from_data_dict("phi1", data), [1, 3])
    assert np.allclose(get_coord_from_data_dict("phi2", data), [2, 4])
