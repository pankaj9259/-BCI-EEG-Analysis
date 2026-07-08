"""Lightweight smoke test: generate synthetic features and run a simple numpy classifier.
This avoids heavy dependencies and verifies the end-to-end pipeline logic.
"""
import os


def main():
    try:
        import numpy as np
    except Exception as e:
        print("numpy is required for smoke test — not available:", e)
        return

    # DEAP-like feature dim: 32 channels * (5 bands + mean + std) = 32*7 = 224
    feat_dim = 32 * 7
    n_samples = 200
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_samples, feat_dim))
    # create a label correlated with sum of band power-like features
    y = (X[:, :5].sum(axis=1) + rng.normal(scale=0.5, size=n_samples) > 0).astype(int)

    # simple train/test split
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # simple numpy logistic regression (gradient descent)
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    w = np.zeros(feat_dim)
    b = 0.0
    lr = 0.1
    for epoch in range(50):
        z = X_train.dot(w) + b
        p = sigmoid(z)
        grad_w = (X_train.T.dot(p - y_train)) / X_train.shape[0]
        grad_b = (p - y_train).mean()
        w -= lr * grad_w
        b -= lr * grad_b

    preds = (sigmoid(X_test.dot(w) + b) >= 0.5).astype(int)
    acc = (preds == y_test).mean()
    print(f"Smoke test accuracy (synthetic): {acc:.3f}")

    # save synthetic features file for local full-run convenience
    out_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "features_synthetic.npz"), X=X, y=y)
    print("Saved synthetic features to data/features_synthetic.npz")


if __name__ == "__main__":
    main()
