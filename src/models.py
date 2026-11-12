"""Text feature engineering and baseline model helpers."""

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def combine_text(dataframe):
    """Combine prompt and both responses with explicit field markers."""
    return (
        "PROMPT " + dataframe["prompt"].fillna("").astype(str)
        + " RESPONSE_A " + dataframe["response_a"].fillna("").astype(str)
        + " RESPONSE_B " + dataframe["response_b"].fillna("").astype(str)
    )


def build_pairwise_features(train_df, validation_df, max_features=100000):
    """Build prompt + (response A - response B) TF-IDF features.

    The vectorizer and scaler are fitted only on the training partition.
    Returns sparse train/validation matrices and fitted transformers.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )
    vectorizer.fit(combine_text(train_df))

    def text_matrix(frame, column):
        return vectorizer.transform(
            frame[column].fillna("").astype(str)
        )

    prompt_train = text_matrix(train_df, "prompt")
    prompt_validation = text_matrix(validation_df, "prompt")
    response_a_train = text_matrix(train_df, "response_a")
    response_b_train = text_matrix(train_df, "response_b")
    response_a_validation = text_matrix(validation_df, "response_a")
    response_b_validation = text_matrix(validation_df, "response_b")

    train_pairwise = hstack([
        prompt_train,
        response_a_train - response_b_train,
    ]).tocsr()
    validation_pairwise = hstack([
        prompt_validation,
        response_a_validation - response_b_validation,
    ]).tocsr()

    def length_difference(frame):
        a_length = frame["response_a"].fillna("").astype(str).str.split().str.len()
        b_length = frame["response_b"].fillna("").astype(str).str.split().str.len()
        return (
            np.log1p(a_length.to_numpy())
            - np.log1p(b_length.to_numpy())
        ).reshape(-1, 1)

    scaler = StandardScaler()
    train_length = scaler.fit_transform(length_difference(train_df))
    validation_length = scaler.transform(length_difference(validation_df))

    train_features = hstack([
        train_pairwise, csr_matrix(train_length)
    ]).tocsr()
    validation_features = hstack([
        validation_pairwise, csr_matrix(validation_length)
    ]).tocsr()

    return train_features, validation_features, vectorizer, scaler


def create_logistic_regression(C=0.1, random_state=42):
    """Create the tuned Logistic Regression classifier."""
    return LogisticRegression(
        C=C,
        max_iter=1000,
        solver="liblinear",
        random_state=random_state,
    )
