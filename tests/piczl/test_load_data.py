import piczl as pc
import pytest


def test_load_data():
    """This is the test for data loader"""
    with pytest.raises(Exception):
        # Check it fails when given 1s
        pc.utilities.data_loader(1,1,1,1)

