import os
import pytest


@pytest.fixture()
def datapath():
    datapath = "data/cifar-10-batches-py"
    assert os.path.exists(datapath)
    return datapath
