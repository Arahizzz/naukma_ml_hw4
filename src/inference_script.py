import click
import joblib
import pandas as pd
from preprocessing import do_preprocessing

class_labels = ["Toxic", "Severe_Toxic", "Obscene", "Threat", "Insult", "Identity_Hate"]


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


@click.command(help="")
@click.option(
    "--model-path",
    default="checkpoints/downloaded_classifier_model.joblib",
    type=str,
    help="local model save path",
)
@click.option(
    "--vectorizer-path",
    default="checkpoints/downloaded_vectorizer.joblib",
    type=str,
    help="local vectorizer save path",
)
def main(model_path, vectorizer_path):
    loaded_clf = joblib.load(model_path)
    loaded_vectorizer = joblib.load(vectorizer_path)

    print(loaded_vectorizer.get_feature_names_out())

    while True:
        input_text = get_input_text()

        processed_text = " ".join(do_preprocessing(input_text))

        print("Processed text: ", processed_text)

        prediction = make_prediction(loaded_clf, loaded_vectorizer, processed_text)

        output_results(prediction)


if __name__ == "__main__":
    main()
