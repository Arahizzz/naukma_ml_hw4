
from src.train import load_and_preprocess_data


def test_load_data(train_dataset_loc, test_dataset_loc):
    df_train, df_test = load_and_preprocess_data(train_dataset_loc, test_dataset_loc)
    assert df_train.shape == (25, 9)
    assert df_test.shape == (25, 2)