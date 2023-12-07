import pandas as pd
import requests
import joblib


class_labels = ["Toxic", "Severe_Toxic", "Obscene", "Threat", "Insult", "Identity_Hate"]


def download_model_and_vectorizer(
    url_model, url_vectorizer, save_path_model, save_path_vectorizer
):
    response_model = requests.get(url_model)
    with open(save_path_model, "wb") as f:
        f.write(response_model.content)

    response_vectorizer = requests.get(url_vectorizer)
    with open(save_path_vectorizer, "wb") as f:
        f.write(response_vectorizer.content)


def get_input_text():
    text = input("Enter the text for prediction: ")
    return text


def make_prediction(model, vectorizer, input_text):
    input_vector = vectorizer.transform([input_text])
    prediction = model.predict_proba(input_vector)
    return prediction


def output_results(prediction_proba):
    prediction_df = pd.DataFrame()
    for i, output_name in enumerate(class_labels):
        prediction_df[output_name] = prediction_proba[i][:, 1]
    formatted_df = prediction_df.round(3).to_string(
        float_format=lambda x: f"{x:.3f}" if 0 < x < 1 else f"{x:.0f}", index=False
    )
    print(f"Predicted classes:\n{formatted_df}")


if __name__ == "__main__":
    # weights_url = "https://raw.githubusercontent.com/username/repo/main/multi_output_classifier_model.joblib"
    # vectorizer_url = "https://raw.githubusercontent.com/username/repo/main/tfidf_vectorizer.joblib"

    local_weights_path = "models/history/linear_regression_classifier_model.joblib"
    local_vectorizer_path = "models/history/tfidf_vectorizer.joblib"

    # download_model_and_vectorizer(weights_url, vectorizer_url, local_weights_path, local_vectorizer_path)

    loaded_clf = joblib.load(local_weights_path)
    loaded_vectorizer = joblib.load(local_vectorizer_path)

    input_text = get_input_text()

    prediction = make_prediction(loaded_clf, loaded_vectorizer, input_text)

    output_results(prediction)
