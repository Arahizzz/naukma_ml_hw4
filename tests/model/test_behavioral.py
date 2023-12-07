import pytest
from numpy.testing import assert_array_equal


@pytest.mark.parametrize(
    "input",
    [
        "Green tree",
        "Blue sky",
        "Fool",
        "Horrible article",
        "Kill yourself",
        "I hate you",
    ],
)
def test_mft(input, predictor, vectorizer):
    """Minimum Functionality Tests (simple input/output pairs)"""
    prediction = get_label(text=input, predictor=predictor, vectorizer=vectorizer)

    # Prediction should be a numpy array of floats between 0 and 1
    assert all([0 <= p <= 1 for p in prediction])

    # Prediction length should be equal to the number of classes
    assert len(prediction) == 6


@pytest.mark.parametrize(
    "inp, is_bad",
    [("Nice cat", False), ("Good dog", False), ("Idiot", True), ("Stupid", True)],
)
def test_sentiment(inp, is_bad, predictor, vectorizer):
    """Sentiment (good/bad)"""
    label = get_label(text=inp, predictor=predictor, vectorizer=vectorizer)

    if is_bad:
        # At least one of the labels should be non-zero
        assert sum(label) > 0
    else:
        # All labels should be zero
        assert sum(label) == 0


@pytest.mark.parametrize(
    "inp_a, inp_b", [("Nice cat", "Good dog"), ("Idiot", "Stupid")]
)
def test_invariance(inp_a, inp_b, predictor, vectorizer):
    """Invariance (changes should not affect outputs)"""
    label_a = get_label(text=inp_a, predictor=predictor, vectorizer=vectorizer)
    label_b = get_label(text=inp_b, predictor=predictor, vectorizer=vectorizer)
    assert_array_equal(label_a, label_b)


def get_label(text, predictor, vectorizer):
    vect = vectorizer.transform([text])
    label = predictor.predict(vect)[0]
    return label
