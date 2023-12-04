import joblib
import pandas as pd
from sklearn.base import accuracy_score
from sklearn.feature_extraction import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from preprocessing import do_preprocessing

columns_base = ['ID', 'Comment_Text']
columns_type = ['Is_Toxic', 'Is_Severe_Toxic', 'Is_Obscene', 'Is_Threat', 'Is_Insult', 'Is_Identity_Hate']
class_labels = ['Toxic', 'Severe_Toxic', 'Obscene', 'Threat', 'Insult', 'Identity_Hate']
columns_all = columns_base + columns_type

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path, usecols=['id', 'comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    test_df = pd.read_csv(test_path, usecols=['id', 'comment_text'])

    # Rename columns in the DataFrame
    train_df.columns = columns_all
    test_df.columns = columns_base

    # Preprocessing text data
    train_df_copy = train_df.copy()
    train_df_copy['Comment_Text_Preprocessed'] = train_df_copy["Comment_Text"].apply(lambda d: " ".join(do_preprocessing(d)))

    return train_df_copy, test_df

def init_model(type):
    if (type == "logistic_regression"):
        clf = LogisticRegression(max_iter=1000)
    elif (type == "random_forest"):
        clf = RandomForestClassifier(random_state=42)
    else: 
        raise ValueError(f"Unknown model type provided - {type}")
    moc = MultiOutputClassifier(clf)
    return moc


def init_vectorizer(X):
    tfidf_vectorizer = TfidfVectorizer(max_features=10_000, max_df=0.9, smooth_idf=True, use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(X)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    return tfidf_vectorizer, feature_names, tfidf_df

def train_model(train_df, type):
    vectorizer, _, _ = init_vectorizer(train_df.toarray())

    # Filter all hate comments for model training
    hate_comments_df = train_df[train_df[columns_type].any(axis=1)].copy().reset_index(drop=True)

    # Filter the same amount of good comments for model training
    good_comments_df = train_df[train_df[columns_type].eq(0).all(axis=1)].sample(n=len(hate_comments_df), random_state=42).copy().reset_index(drop=True)

    # Concatenate 50% hate and 50% good comments and shuffle
    train_df_balanced = pd.concat([hate_comments_df, good_comments_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    X = train_df_balanced['Comment_Text_Preprocessed']
    y = train_df_balanced[columns_type]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    model = init_model(type)
    
    model.fit(X_train_vectorized, y_train)
    
    print('Accuracy Score: ', accuracy_score(y_test, model.predict(X_test_vectorized)))
    # Save model
    joblib.dump(model, f'../models/{type}_classifier_model.joblib')

