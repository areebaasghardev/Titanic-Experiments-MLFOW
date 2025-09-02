import pandas as pd
import yaml
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

def train(params):
    # Load preprocessed CSVs
    X_train = pd.read_csv(params["preprocess"]["output_X_train"])
    y_train = pd.read_csv(params["preprocess"]["output_y_train"])

    # Ensure only numeric columns are used
    X_train = X_train.select_dtypes(include=["number"])

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

    # Split into train/validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train,
        test_size=params["train"]["test_size"],
        random_state=params["train"]["random_state"]
    )

    # MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # local MLflow server
    mlflow.set_experiment("Titanic-Experiments")

    with mlflow.start_run():
        # Train model
        model = LogisticRegression(max_iter=params["train"]["max_iter"])
        model.fit(X_train_split, y_train_split)

        # Evaluate
        preds = model.predict(X_val)
        acc = accuracy_score(y_val, preds)

        # Log params, metrics, and model
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", params["train"]["max_iter"])
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"Validation Accuracy: {acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", default="params.yaml")
    args = parser.parse_args()
    
    with open(args.params) as f:
        params = yaml.safe_load(f)
    
    train(params)
