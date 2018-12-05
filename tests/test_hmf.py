import pytest
from hmf import *
import numpy as np
import numpy.testing as npt

def test_hmf_emulator_load_data():
    h = hmf.hmf_emulator()
    npt.assert_equal(h.loaded_data, True)
    attrs = ["data_path", "training_cosmologies",
             "rotation_matrix", "training_data",
             "training_mean", "training_stddev"]
    for attr in attrs:
        npt.assert_equal(hasattr(h, attr), True)
