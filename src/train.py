import os
import click
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

from preprocessing import do_preprocessing

columns_base = ['ID', 'Comment_Text']
columns_type = ['Is_Toxic', 'Is_Severe_Toxic', 'Is_Obscene', 'Is_Threat', 'Is_Insult', 'Is_Identity_Hate']
class_labels = ['Toxic', 'Severe_Toxic', 'Obscene', 'Threat', 'Insult', 'Identity_Hate']
columns_all = columns_base + columns_type

def load_and_preprocess_data(train_path, test_path=None):
    print(f"Loading data from {train_path}")
    train_df = pd.read_csv(train_path, usecols=['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    test_df = pd.read_csv(test_path, usecols=['id', 'comment_text']) if test_path else None

    # Rename columns in the DataFrame
    train_df.columns = columns_all

    # Preprocessing text data
    print(f"Preprocessing data")
    train_df_preprocessed = train_df.copy()
    train_df_preprocessed['Comment_Text_Preprocessed'] = train_df_preprocessed["Comment_Text"].parallel_apply(lambda d: " ".join(do_preprocessing(d)))

    return train_df_preprocessed, test_df

def init_model(type):
    print(f"Training model {type}")
    if (type == "logistic_regression"):
        clf = LogisticRegression(max_iter=1000)
    elif (type == "random_forest"):
        clf = RandomForestClassifier(random_state=42)
    else: 
        raise ValueError(f"Unknown model type provided - {type}")
    moc = MultiOutputClassifier(clf)
    return moc


def init_vectorizer(X):
    print(f"Preparing vectorizer")
    tfidf_vectorizer = TfidfVectorizer(max_features=10_000, max_df=0.9, smooth_idf=True, use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(X)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    return tfidf_vectorizer, feature_names, tfidf_df

def train_model(train_df, vectorizer, type, out_folder):

    # Filter all hate comments for model training
    hate_comments_df = train_df[train_df[columns_type].any(axis=1)].copy().reset_index(drop=True)

    # Filter the same amount of good comments for model training
    good_comments_df = train_df[train_df[columns_type].eq(0).all(axis=1)].sample(n=len(hate_comments_df), random_state=42).copy().reset_index(drop=True)

    # Concatenate 50% hate and 50% good comments and shuffle
    train_df_balanced = pd.concat([hate_comments_df, good_comments_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    X = train_df_balanced['Comment_Text_Preprocessed']
    y = train_df_balanced[columns_type]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    print(f"Vectorizing data")
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = init_model(type)
    
    print(f"Fitting model")
    model.fit(X_train_vectorized, y_train)
    
    print('Accuracy Score: ', accuracy_score(y_test, model.predict(X_test_vectorized)))

    return model



@click.command(help="")
@click.option("--model-type", default="logistic_regression", type=str, help="model name")
@click.option("--data-folder", default="input", type=str, help="input data folder")
@click.option("--output-folder", default="checkpoints", type=str, help="output models folder")
@click.option("--start-step", default="preprocess", type=str, help="start from step (preprocess, vectorize, train)")
def main(model_type, data_folder, output_folder, start_step):
    if start_step == "preprocess":
        train_df, _ = load_and_preprocess_data(os.path.join(data_folder, 'train.csv'))
        # Save preprocessed data
        joblib.dump(train_df, f'{output_folder}/train_df_preprocessed.joblib')
        print(f"Preprocessed data saved to {output_folder}/train_df_preprocessed.joblib")
    else:
        train_df = joblib.load(f'{output_folder}/train_df_preprocessed.joblib')
        print(f"Preprocessed data loaded from {output_folder}/train_df_preprocessed.joblib")
    

    if start_step == "vectorize" or start_step == "preprocess":
        vectorizer, _, _ = init_vectorizer(train_df)
        # Save vectorizer
        joblib.dump(vectorizer, f'{output_folder}/tfidf_vectorizer.joblib')
        print(f"Vectorizer saved to {output_folder}/tfidf_vectorizer.joblib")
    else:
        vectorizer = joblib.load(f'{output_folder}/tfidf_vectorizer.joblib')
        print(f"Vectorizer loaded from {output_folder}/tfidf_vectorizer.joblib")

    model = train_model(train_df, vectorizer, model_type, output_folder)
    # Save model
    joblib.dump(model, f'{output_folder}/{model_type}_classifier_model.joblib')
    print(f"Model saved to {output_folder}/{model_type}_classifier_model.joblib")

if __name__ == "__main__":
    main()