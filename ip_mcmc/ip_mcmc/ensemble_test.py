import pytest

import numpy as np

from tempfile import TemporaryDirectory

from .ensemble import NumpyDataManager, EnsembleManager


@pytest.fixture
def tempdir():
    # Regardless of test result, the code after
    # yield will be exectuted (in particular,
    # data_dir goes out of scope and gets cleaned up)
    with TemporaryDirectory() as data_dir:
        yield data_dir


def test_NumpyDataManager(tempdir):
    m = NumpyDataManager(tempdir, "testing")

    assert m.data_exists() is False, ""

    d = np.random.rand(10, 10, 10)
    m.store(d)

    assert m.data_exists() is True, ""
    assert np.array_equal(m.load(), d), ""


def test_EnsembleManager(tempdir):
    m = EnsembleManager(tempdir, "testing")

    a = [1, 2, 3, 4, 5]

    res = m.compute(np.sqrt, a, 5)

    # Should actually check whether the second time the
    # result is loaded rather than recomputet. For now: just
    # read the print statement.
    # To automate: set up EnsembleManager so that it's testable
    # or read stdout
    assert np.array_equal(res, m.compute(np.sqrt, a, 5)), ""
