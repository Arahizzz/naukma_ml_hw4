import pytest


@pytest.fixture
def train_dataset_loc():
    return "tests/fixtures/sample_data_train.csv"


@pytest.fixture
def test_dataset_loc():
    return "tests/fixtures/sample_data_test.csv"
