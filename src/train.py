"""Train ML and ANN-style models on extracted features.

Usage:
    python src/train.py --features data/features.npz --out results
"""
import argparse
import json
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def train_rf(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


def train_ann(X_train, y_train):
    # Fast ANN-style classifier implemented with scikit-learn MLP.
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=64,
        learning_rate_init=1e-3,
        max_iter=100,
        early_stopping=True,
        n_iter_no_change=10,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    data = np.load(args.features)
    X = data['X']
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    os.makedirs(args.out, exist_ok=True)

    print("Training RandomForest...")
    rf = train_rf(X_train, y_train)
    joblib.dump(rf, os.path.join(args.out, 'rf.joblib'))
    y_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    rf_f1 = f1_score(y_test, y_pred)
    rf_report = classification_report(y_test, y_pred)
    print("RandomForest test acc:", rf_acc)
    print(rf_report)

    print("Training ANN-style MLP...")
    ann = train_ann(X_train, y_train)
    joblib.dump(ann, os.path.join(args.out, 'ann_mlp.joblib'))
    y_pred_ann = ann.predict(X_test)
    ann_acc = accuracy_score(y_test, y_pred_ann)
    ann_f1 = f1_score(y_test, y_pred_ann)
    ann_report = classification_report(y_test, y_pred_ann)
    print("ANN test acc:", ann_acc)
    print(ann_report)

    metrics = {
        "random_forest": {"accuracy": rf_acc, "f1": rf_f1, "report": rf_report},
        "ann_mlp": {"accuracy": ann_acc, "f1": ann_f1, "report": ann_report},
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(args.out, "report.txt"), "w", encoding="utf-8") as f:
        f.write("RandomForest\n")
        f.write(rf_report)
        f.write("\n\nANN-Style MLP\n")
        f.write(ann_report)


if __name__ == "__main__":
    main()
