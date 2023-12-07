from src.preprocessing import (
    convert_nums_to_words,
    do_lemmatization,
    do_preprocessing,
    do_stemming,
    remove_long_dash,
    remove_numeric_words,
    remove_one_letter_words,
    remove_punctuation,
    remove_stop_words,
    remove_urls,
    to_lower_case,
    tokenize_text,
)


def test_to_lower_case():
    assert to_lower_case("Hello World") == "hello world"


def test_remove_punctuation():
    assert remove_punctuation("Hello, World!") == "Hello World"


def test_remove_long_dash():
    assert remove_long_dash("text—example") == "text example"


def test_remove_urls():
    assert remove_urls("Visit our website: http://example.com") == "Visit our website: "


def test_remove_one_letter_words():
    input_tokens = ["hello", "world", "a", "is", "the"]
    expected_output = ["hello", "world", "is", "the"]
    assert remove_one_letter_words(input_tokens) == expected_output


def test_tokenize_text():
    input_text = "Hello world! This is a test."
    expected_output = ["Hello", "world", "!", "This", "is", "a", "test", "."]
    assert tokenize_text(input_text) == expected_output


def test_remove_stop_words():
    input_tokens = ["This", "is", "a", "test", "sentence"]
    expected_output = ["This", "test", "sentence"]
    assert remove_stop_words(input_tokens) == expected_output


def test_do_stemming():
    input_tokens = ["running", "jumps", "played"]
    expected_output = ["run", "jump", "play"]
    assert do_stemming(input_tokens) == expected_output


def test_do_lemmatization():
    input_tokens = ["running", "jumps", "played"]
    expected_output = ["running", "jump", "played"]
    assert do_lemmatization(input_tokens) == expected_output


def test_remove_numeric_words():
    input_text = "The price is $100"
    expected_output = "The price is "
    assert remove_numeric_words(input_text) == expected_output


def test_convert_nums_to_words():
    input_tokens = ["12", "45"]
    expected_output = ["twelve", "forty", "five"]
    assert convert_nums_to_words(input_tokens) == expected_output


def test_do_preprocessing():
    input_text = (
        """
    == Final draft maybe == 
    This will go on the evolutionary ethics page. https://en.wikipedia.org/wiki/Evolutionary_ethics, it will go after the section called """
        "Further reading"
        """"
    """
    )
    print(do_preprocessing(input_text))
    expected_output = [
        "final",
        "draft",
        "maybe",
        "go",
        "evolutionary",
        "ethic",
        "page",
        "go",
        "section",
        "called",
        "reading",
    ]
    assert do_preprocessing(input_text) == expected_output
