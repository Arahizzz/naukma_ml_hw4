import great_expectations as ge
import pandas as pd
import pytest


def pytest_addoption(parser):
    """Add option to specify dataset location when executing tests from CLI.
    Ex: pytest --dataset-loc=checkpoints/data.csv tests/data --verbose --disable-warnings
    """
    parser.addoption(
        "--test-dataset-loc",
        action="store",
        default=None,
        help="Location of the test dataset.",
    )
    parser.addoption(
        "--train-dataset-loc",
        action="store",
        default=None,
        help="Location of the training dataset.",
    )


@pytest.fixture(scope="module")
def test_df(request, test_dataset_loc):
    """Fixture for loading datasets."""
    test_dataset_location = request.config.getoption("--test-dataset-loc")

    if test_dataset_location is not None:
        dataset_loc = test_dataset_location
    else:
        dataset_loc = test_dataset_loc

    df = ge.dataset.PandasDataset(pd.read_csv(dataset_loc))
    return df


@pytest.fixture(scope="module")
def train_df(request, train_dataset_loc):
    """Fixture for loading datasets."""
    train_dataset_location = request.config.getoption("--train-dataset-loc")

    if train_dataset_location is not None:
        dataset_loc = train_dataset_location
    else:
        dataset_loc = train_dataset_loc

    df = ge.dataset.PandasDataset(pd.read_csv(dataset_loc))
    return df


@pytest.fixture(scope="module")
def train_dataset_loc():
    return "tests/fixtures/sample_data_train.csv"


@pytest.fixture(scope="module")
def test_dataset_loc():
    return "tests/fixtures/sample_data_test.csv"
