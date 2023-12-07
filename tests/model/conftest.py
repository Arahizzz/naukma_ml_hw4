import pytest
from joblib import load


@pytest.fixture(
    scope="module",
    params=[
        "checkpoints/logistic_regression_classifier_model.joblib",
        "checkpoints/random_forest_classifier_model.joblib",
    ],
)
def predictor(request):
    model = load(request.param)
    return model


@pytest.fixture(scope="module")
def vectorizer(request):
    vectorizer = load("checkpoints/tfidf_vectorizer.joblib")
    return vectorizer
