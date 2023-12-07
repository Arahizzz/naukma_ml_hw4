import click
import requests


@click.command(help="")
@click.option("--model-url", type=str, help="model remote url", required=False)
@click.option(
    "--model-path",
    default="checkpoints/downloaded_classifier_model.joblib",
    type=str,
    help="local model save path",
)
@click.option(
    "--vectorizer-url", type=str, help="vectorizer remote url", required=False
)
@click.option(
    "--vectorizer-path",
    default="checkpoints/downloaded_vectorizer.joblib",
    type=str,
    help="local vectorizer save path",
)
def main(model_url, model_path, vectorizer_url, vectorizer_path):
    if model_url is not None:
        print(f"Downloading model from {model_url} to {model_path}")
        response_model = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response_model.content)

    if vectorizer_url is not None:
        print(f"Downloading vectorizer from {vectorizer_url} to {vectorizer_path}")
        response_vectorizer = requests.get(vectorizer_url)
        with open(vectorizer_path, "wb") as f:
            f.write(response_vectorizer.content)


if __name__ == "__main__":
    main()
