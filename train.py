import os
import urllib.request
import zipfile
import io
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

def train_and_save():
    print("Fetching Adult Income dataset from UCI Repository...")
    url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        zf = zipfile.ZipFile(io.BytesIO(resp.read()))
        data_bytes = zf.read("adult.data")
        test_bytes = zf.read("adult.test")

    columns = [
        "age", "workclass", "fnlwgt", "education", "educational-num",
        "marital-status", "occupation", "relationship", "race", "gender",
        "capital-gain", "capital-loss", "hours-per-week", "native-country", "income_>50K"
    ]

    df1 = pd.read_csv(io.BytesIO(data_bytes), header=None, names=columns, skipinitialspace=True)
    df2 = pd.read_csv(io.BytesIO(test_bytes), header=None, names=columns, skipinitialspace=True, skiprows=1)
    df = pd.concat([df1, df2], ignore_index=True)

    print(f"Loaded {len(df)} raw records. Cleaning data...")
    df.replace("?", np.nan, inplace=True)
    df.drop_duplicates(inplace=True, ignore_index=True)
    df["income_>50K"] = df["income_>50K"].str.rstrip(".").map({"<=50K": 0, ">50K": 1})
    df.dropna(inplace=True)

    # Feature Engineering: Region grouping (matching notebook)
    north_america = ["United-States", "Mexico", "Canada", "Puerto-Rico", "Cuba"]
    asia = ["Japan", "South", "Philippines", "China", "Vietnam", "Thailand", "India", "Cambodia", "Yugoslavia", "Iran", "Taiwan", "Hong"]
    europe = ["Portugal", "Italy", "England", "Germany", "Poland", "Greece", "Ireland", "Scotland", "France", "Hungary"]
    latin_america = ["Ecuador", "Dominican-Republic", "Jamaica", "El-Salvador", "Guatemala", "Columbia", "Haiti", "Peru", "Nicaragua", "Trinadad&Tobago"]

    df["Region"] = df["native-country"].apply(
        lambda x: "North America" if x in north_america
        else "Asia" if x in asia
        else "Europe" if x in europe
        else "Latin America" if x in latin_america
        else "Other"
    )

    cat_cols = ["workclass", "marital-status", "occupation", "relationship", "race", "gender", "native-country", "Region"]
    num_cols = ["age", "fnlwgt", "educational-num", "capital-gain", "capital-loss", "hours-per-week"]

    # Drop redundant column 'education' since 'educational-num' is numerical representation
    df_encoded = pd.get_dummies(data=df.drop(columns=["education"]), columns=cat_cols, dtype=int)
    feature_names = [col for col in df_encoded.columns if col != "income_>50K"]

    X = df_encoded[feature_names]
    y = df_encoded["income_>50K"]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=10)

    scaler = StandardScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_test_sc = scaler.transform(x_test)

    print("Training models...")
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(x_train_sc, y_train)
    lr_acc = accuracy_score(y_test, lr_model.predict(x_test_sc))

    ridge_model = RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=42)
    ridge_model.fit(x_train_sc, y_train)
    ridge_acc = accuracy_score(y_test, ridge_model.predict(x_test_sc))

    nb_model = GaussianNB()
    nb_model.fit(x_train_sc, y_train)
    nb_acc = accuracy_score(y_test, nb_model.predict(x_test_sc))

    print(f"Results -> Logistic: {lr_acc:.4f}, Ridge: {ridge_acc:.4f}, Naive Bayes: {nb_acc:.4f}")

    os.makedirs("model", exist_ok=True)
    bundle = {
        "scaler": scaler,
        "model": lr_model,
        "ridge_model": ridge_model,
        "nb_model": nb_model,
        "accuracies": {
            "Logistic Regression": float(lr_acc),
            "Ridge Classifier": float(ridge_acc),
            "Gaussian Naive Bayes": float(nb_acc)
        },
        "feature_names": feature_names,
        "categorical_cols": cat_cols,
        "numerical_cols": num_cols,
        "categories_dict": {col: sorted(df[col].unique().tolist()) for col in cat_cols}
    }

    with open("model/model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Model bundle exported to model/model_bundle.pkl")

if __name__ == "__main__":
    train_and_save()
