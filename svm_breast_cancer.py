"""
Support Vector Machine for Breast Cancer Classification
=======================================================

Goal: classify breast tumors as malignant or benign using the
Breast Cancer Wisconsin (Diagnostic) dataset.

What this file does, in order:
  Part 1. Defines our LinearSVM class, written from scratch with NumPy.
  Part 2. Loads the dataset and converts the labels to -1 / +1.
  Part 3. Splits the data into a training set and a test set.
  Part 4. Scales the features so they are on comparable ranges.
  Part 5. Trains our LinearSVM and plots its loss curve.
  Part 6. Trains scikit-learn's SVC with the same settings to check our work.
  Part 7. Trains an RBF-kernel SVC to check whether the data needs a
          nonlinear boundary.
  Part 8. Evaluates all three models on the test set.

Run with:  python svm_breast_cancer.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)


# =============================================================================
# PART 1: OUR LINEAR SVM
# =============================================================================

class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C            # penalty for margin violations
        self.lr = lr          # learning rate: how big each gradient step is
        self.epochs = epochs  # how many gradient steps to take

    def fit(self, X, y):
        """Train the model. X is (n_samples, n_features); y holds -1 or +1."""
        n_features = X.shape[1]
        self.w = np.zeros(n_features)  # one weight per feature, start at 0
        self.b = 0.0
        self.losses = []               # objective value after each epoch

        for _ in range(self.epochs):
            # Step 1: y * f(x) for every sample
            margins = y * (X @ self.w + self.b)

            # Step 2: True for samples inside the margin or misclassified
            violators = margins < 1

            # Step 3: gradient of the objective
            dw = self.w - self.C * (y[violators] @ X[violators])
            db = -self.C * y[violators].sum()

            # Step 4: move w and b against the gradient
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Record the objective so we can plot that training converges
            hinge = np.maximum(0, 1 - margins).sum()
            self.losses.append(0.5 * self.w @ self.w + self.C * hinge)

        return self

    def decision_function(self, X):
        """Score f(x) = w . x + b for each sample."""
        return X @ self.w + self.b

    def predict(self, X):
        """+1 (benign) if the score is >= 0, otherwise -1 (malignant)."""
        return np.where(self.decision_function(X) >= 0, 1, -1)


# =============================================================================
# SETTINGS
# =============================================================================

RANDOM_STATE = 42     # fixed seed so the train/test split is the same every run
C = 1.0               # SVM penalty, shared by our model and sklearn's linear SVC
LEARNING_RATE = 0.0005
EPOCHS = 5000

os.makedirs("figures", exist_ok=True)


# =============================================================================
# PART 2: LOAD THE DATA
# =============================================================================

print("=" * 60)
print("PART 2: Load data")
print("=" * 60)

data = load_breast_cancer()
X = data.data
y = np.where(data.target == 0, -1, 1)

print(f"Samples: {X.shape[0]}, features: {X.shape[1]}")
print(f"Malignant (-1): {(y == -1).sum()}, benign (+1): {(y == 1).sum()}")


# =============================================================================
# PART 3: TRAIN/TEST SPLIT
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

print(f"Training samples: {len(y_train)}, test samples: {len(y_test)}")


# =============================================================================
# PART 4: SCALE THE FEATURES
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: Scale features")
print("=" * 60)

print(f"Before scaling, mean smoothness ranges "
      f"{X[:, 4].min():.3f} to {X[:, 4].max():.3f}")
print(f"Before scaling, worst area ranges "
      f"{X[:, 23].min():.0f} to {X[:, 23].max():.0f}")

scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print("After scaling, every feature has mean 0 and standard deviation 1 "
      "(on the training set)")


# =============================================================================
# PART 5: TRAIN OUR LINEAR SVM
# =============================================================================

print("\n" + "=" * 60)
print("PART 5: Train our LinearSVM")
print("=" * 60)

ours = LinearSVM(C=C, lr=LEARNING_RATE, epochs=EPOCHS)
ours.fit(X_train, y_train)

print(f"Objective at epoch 1:    {ours.losses[0]:.2f}")
print(f"Objective at epoch {EPOCHS}: {ours.losses[-1]:.2f}")

plt.figure(figsize=(6, 4))
plt.plot(ours.losses)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Objective (log scale)")
plt.title("Training loss: our LinearSVM")
plt.tight_layout()
plt.savefig("figures/loss_curve.png", dpi=150)

# Which features matter most? Larger |w| means more influence on the score.
# A negative weight pushes a sample toward malignant (-1).
print("\nTop 5 features by |weight| (negative pushes toward malignant):")
for i in np.argsort(np.abs(ours.w))[::-1][:5]:
    print(f"  {data.feature_names[i]:<25} {ours.w[i]: .3f}")


# =============================================================================
# PART 6: CHECK OUR MODEL AGAINST SCIKIT-LEARN
# =============================================================================

print("\n" + "=" * 60)
print("PART 6: Compare with scikit-learn's linear SVC")
print("=" * 60)

sk_linear = SVC(kernel="linear", C=C).fit(X_train, y_train)
w_sk = sk_linear.coef_[0]
b_sk = sk_linear.intercept_[0]

sk_objective = (0.5 * w_sk @ w_sk
                + C * np.maximum(0, 1 - y_train * (X_train @ w_sk + b_sk)).sum())
cos_sim = (ours.w @ w_sk) / (np.linalg.norm(ours.w) * np.linalg.norm(w_sk))

print(f"Final objective:  ours {ours.losses[-1]:.3f},  sklearn {sk_objective:.3f}")
print(f"Bias b:           ours {ours.b:.3f},  sklearn {b_sk:.3f}")
print(f"Cosine similarity of weight vectors: {cos_sim:.4f}")


# =============================================================================
# PART 7: CHECKING A NONLINEAR BOUNDARY
# =============================================================================

print("\n" + "=" * 60)
print("PART 7: Tune an RBF-kernel SVC (nonlinearity check)")
print("=" * 60)

param_grid = {"C": [0.1, 1, 10, 100], "gamma": ["scale", 0.001, 0.01, 0.1]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)
sk_rbf = grid.best_estimator_

print(f"Best settings: {grid.best_params_}")
print(f"Cross-validation accuracy: {grid.best_score_:.3f}")


# =============================================================================
# PART 8: EVALUATE ON THE TEST SET
# =============================================================================

print("\n" + "=" * 60)
print(f"PART 8: Test set results ({len(y_test)} samples, "
      "positive class = malignant)")
print("=" * 60)

models = {
    "Ours (linear)": ours,
    "sklearn linear": sk_linear,
    "sklearn RBF": sk_rbf,
}

print(f"{'Model':<16}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, (name, model) in zip(axes, models.items()):
    pred = model.predict(X_test)
    print(f"{name:<16}"
          f"{accuracy_score(y_test, pred):>10.3f}"
          f"{precision_score(y_test, pred, pos_label=-1):>11.3f}"
          f"{recall_score(y_test, pred, pos_label=-1):>9.3f}"
          f"{f1_score(y_test, pred, pos_label=-1):>8.3f}")

    cm = confusion_matrix(y_test, pred, labels=[-1, 1])
    ConfusionMatrixDisplay(cm, display_labels=["malignant", "benign"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)

plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)

print("\nPlots saved to the figures/ folder.")
plt.show()
