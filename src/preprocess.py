#!/usr/bin/env python3
"""
Preprocess Titanic raw data into processed CSVs:
 - data/train.csv
 - data/test.csv
 - data/gender_submission.csv (optional)

Outputs:
 - data/X_train.csv
 - data/y_train.csv
 - data/X_test.csv
"""

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os

RAW_DIR = os.environ.get("RAW_DIR", "data")
OUT_DIR = os.environ.get("OUT_DIR", "data")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def preprocess_train(train_df: pd.DataFrame):
    df = train_df.copy()

    # Drop columns
    df = df.drop(columns=["Ticket", "PassengerId", "Cabin"])

    # Encode Sex & Embarked
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})
    df["Embarked"] = df["Embarked"].fillna(2)

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Title
    df["Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
    df = df.drop(columns="Name")
    df["Title"] = df["Title"].replace(
        ["Dr", "Rev", "Col", "Major", "Countess", "Sir", "Jonkheer", "Lady", "Capt", "Don"], "Others"
    )
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")
    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].map({"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Others": 4})

    # Impute Age
    nan_indexes = df["Age"][df["Age"].isnull()].index
    for i in nan_indexes:
        pred_age = df["Age"][
            ((df.SibSp == df.iloc[i]["SibSp"]) &
             (df.Parch == df.iloc[i]["Parch"]) &
             (df.Pclass == df.iloc[i]["Pclass"]))
        ].median()
        if not np.isnan(pred_age):
            df.loc[i, "Age"] = pred_age
        else:
            df.loc[i, "Age"] = df["Age"].median()

    return df


def preprocess_test(test_df: pd.DataFrame, train_df: pd.DataFrame):
    df = test_df.copy()
    df = df.drop(columns=["Ticket", "PassengerId", "Cabin"])

    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df["Embarked"] = df["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    df["Title"] = df.Name.str.extract(" ([A-Za-z]+)\.", expand=False)
    df = df.drop(columns="Name")
    df["Title"] = df["Title"].replace(
        ["Dr", "Rev", "Col", "Major", "Countess", "Sir", "Jonkheer", "Lady", "Capt", "Don"], "Others"
    )
    df["Title"] = df["Title"].replace("Ms", "Miss")
    df["Title"] = df["Title"].replace("Mme", "Mrs")
    df["Title"] = df["Title"].replace("Mlle", "Miss")
    df["Title"] = df["Title"].map({"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Others": 4})

    # Impute Age using train_df statistics when possible
    nan_indexes = df["Age"][df["Age"].isnull()].index
    for i in nan_indexes:
        # use similar rows in train_df
        pred_age = train_df["Age"][
            ((train_df.SibSp == df.iloc[i]["SibSp"]) &
             (train_df.Parch == df.iloc[i]["Parch"]) &
             (train_df.Pclass == df.iloc[i]["Pclass"]))
        ].median()
        if not np.isnan(pred_age):
            df.loc[i, "Age"] = pred_age
        else:
            df.loc[i, "Age"] = train_df["Age"].median()

    # Fill Title and Fare missing
    title_mode = train_df.Title.mode()[0]
    df.Title = df.Title.fillna(title_mode)
    fare_mean = train_df.Fare.mean()
    df.Fare = df.Fare.fillna(fare_mean)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    return df


def main():
    ensure_dir(OUT_DIR)

    train_path = os.path.join(RAW_DIR, "train.csv")
    test_path = os.path.join(RAW_DIR, "test.csv")
    gender_sub_path = os.path.join(RAW_DIR, "gender_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # gender_submission may be unused for preprocess but kept if present
    try:
        _ = pd.read_csv(gender_sub_path)
    except Exception:
        pass

    train_processed = preprocess_train(train_df)
    test_processed = preprocess_test(test_df, train_processed)

    # Shuffle and split
    train_processed = shuffle(train_processed, random_state=42)

    X_train = train_processed.drop(columns="Survived")
    y_train = pd.DataFrame({"Survived": train_processed.Survived.values})
    X_test = test_processed

    X_train.to_csv(os.path.join(OUT_DIR, "X_train.csv"), index=False)
    y_train.to_csv(os.path.join(OUT_DIR, "y_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUT_DIR, "X_test.csv"), index=False)

    print("Preprocessing done. Files written to", OUT_DIR)


if __name__ == "__main__":
    main()
