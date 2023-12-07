# Machine Learning Homework 4

<!--TOC-->

- [Machine Learning Homework 4](#machine-learning-homework-4)
  - [Getting Started](#getting-started)
    - [Via Docker (easier)](#via-docker-easier)
    - [Via Poetry (for development)](#via-poetry-for-development)
  - [Project Structure](#project-structure)
  - [Continuous Integration](#continuous-integration)

<!--TOC-->

## Getting Started

### Via Docker (easier)

1. Install [Docker](https://docs.docker.com/get-docker/)
1. Clone this repo
1. Build the docker image:

```sh
docker build -t ml-hw4 -f docker/Dockerfile \
--build-arg="MODEL_URI=https://github.com/brutovsky/naukma_ml_hw4/raw/main/checkpoints/logistic_regression_classifier_model.joblib?download=" \
--build-arg="VECTORIZER_URI=https://github.com/brutovsky/naukma_ml_hw4/raw/main/checkpoints/tfidf_vectorizer.joblib?download=" \
.
```

4. Run the inference script directly:

```sh
docker run --rm -it ml-hw4
```

5. Enter the container environment to run other commands:

```sh
docker run --rm -it ml-hw4 bash
```

### Via Poetry (for development)

1. Install [Poetry](https://python-poetry.org/docs/#installation)
1. Clone this repo
1. Install dependencies:

```sh
poetry install
```

4. Enter the environment:

```sh
poetry shell
```

5. Setup the pre-commit hooks:

```sh
pre-commit install
```

## Project Structure

- EDA - notebooks with exploratory data analysis
- checkpoints - model weights, vectorizer and other checkpoints
- input - input data (CSV)
- docker
- src - source code
  - `train.py` - training CLI
  - `inference_script.py` - inference CLI
  - `download_model.py` - tool for downloading model and vectorizer
  - `preprocessing.py` - library module with preprocessing functions
- tests
  - model - tests for model behavior
  - code - tests for training functions
  - fixtures - test fixtures (sample data, etc.)

## Continuous Integration

Project is linted and tested on every push using GitHub Actions.\
For tests code [pytest](https://docs.pytest.org/en/6.2.x/) is used. Code coverage is measured using [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) and is mandated to be at least 40%.
Code is formatted on each commit using [pre-commit hooks](https://pre-commit.com/).\
For the linting and formatting [ruff library](https://github.com/astral-sh/ruff) is used.
