def test_test_dataset(test_df):
    """Test test dataset quality and integrity."""
    check_comments_column(test_df)
    # Expectation suite
    expectation_suite = test_df.get_expectation_suite(discard_failed_expectations=False)
    results = test_df.validate(
        expectation_suite=expectation_suite, only_return_failures=True
    ).to_json_dict()
    assert results["success"]


def test_train_dataset(train_df):
    """Test train dataset quality and integrity."""
    columns = [
        "id",
        "comment_text",
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    ]
    train_df.expect_table_columns_to_match_ordered_list(column_list=columns)
    train_df.expect_column_values_to_be_in_set(column="toxic", value_set=[0, 1])
    train_df.expect_column_values_to_be_in_set(column="severe_toxic", value_set=[0, 1])
    train_df.expect_column_values_to_be_in_set(column="obscene", value_set=[0, 1])
    train_df.expect_column_values_to_be_in_set(column="threat", value_set=[0, 1])
    train_df.expect_column_values_to_be_in_set(column="insult", value_set=[0, 1])
    train_df.expect_column_values_to_be_in_set(column="identity_hate", value_set=[0, 1])
    # missing values
    check_comments_column(train_df)
    # Expectation suite
    expectation_suite = train_df.get_expectation_suite(
        discard_failed_expectations=False
    )
    results = train_df.validate(
        expectation_suite=expectation_suite, only_return_failures=True
    ).to_json_dict()
    assert results["success"]


def check_comments_column(df):
    # missing values
    df.expect_column_values_to_not_be_null(column="comment_text")
    df.expect_column_values_to_not_be_null(column="id")
    # unique values
    df.expect_column_values_to_be_unique(column="id")
    # type adherence
    df.expect_column_values_to_be_of_type(column="comment_text", type_="str")
