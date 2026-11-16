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
# PART 1: OUR LINEAR SVM, WRITTEN FROM SCRATCH
# =============================================================================
#
# The model
# ---------
# A linear SVM learns a weight vector w (one weight per feature) and a bias b.
# For a sample x, it computes the score
#
#     f(x) = w . x + b
#
# and predicts benign (+1) if f(x) >= 0, malignant (-1) otherwise.
# The boundary between the classes is the set of points where f(x) = 0.
#
# The objective (what training minimizes)
# ---------------------------------------
#     0.5 * ||w||^2  +  C * sum over all training samples of  max(0, 1 - y * f(x))
#
#   - 0.5 * ||w||^2 keeps w small. The margin width is 2 / ||w||, so a
#     smaller w means a wider margin between the two classes.
#
#   - max(0, 1 - y * f(x)) is the "hinge loss." The value y * f(x) is
#     positive when a sample is on the correct side of the boundary and
#     at least 1 when it is safely outside the margin.
#       * Safely outside the margin  -> loss is 0 (the sample is ignored)
#       * Inside the margin or wrong -> loss grows the further it is off
#
#   - C controls the tradeoff. A large C punishes mistakes heavily (narrow
#     margin, fits training data closely). A small C allows more mistakes
#     in exchange for a wider margin.
#
# How training works (gradient descent)
# -------------------------------------
# Start with w = 0 and b = 0. On every epoch:
#   1. Compute y * f(x) for every training sample.
#   2. Find the "violators": samples with y * f(x) < 1.
#   3. Compute the gradient (the direction that increases the objective):
#        dw = w - C * sum of (y * x) over violators
#        db =   - C * sum of  y      over violators
#   4. Step the other way: w = w - lr * dw,  b = b - lr * db
#
# Samples safely outside the margin do not appear in the gradient at all.
# Only the samples on or inside the margin (the support vectors) shape the
# final boundary.
#
# This is the same objective scikit-learn's SVC minimizes, so if our code is
# correct, both should end up with nearly the same w and b.

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
# The dataset is built into scikit-learn, so no download is needed.
#   X: 569 rows (one per biopsy sample) x 30 columns (features)
#   y: scikit-learn uses 0 = malignant, 1 = benign
#
# The SVM math needs labels of -1 and +1, so we map:
#   0 (malignant) -> -1
#   1 (benign)    -> +1

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
# 80% of samples are used for training and 20% are held back for testing.
# The test set is never seen during training, so it shows how the model
# performs on new patients.
#
# stratify=y keeps the malignant/benign ratio the same in both sets.

print("\n" + "=" * 60)
print("PART 3: Train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

print(f"Training samples: {len(y_train)}, test samples: {len(y_test)}")


# =============================================================================
# PART 4: SCALE THE FEATURES
# =============================================================================
# Features have very different ranges. An SVM measures distances, so without
# scaling, large-valued features like area would dominate small-valued ones
# like smoothness. Gradient descent also converges poorly on unscaled data.
#
# StandardScaler shifts each feature to mean 0 and standard deviation 1.
# The scaler is fit on the training set only, then applied to both sets,
# so no information from the test set leaks into training.

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
# Train our from-scratch model and plot the objective after each epoch.
# A curve that drops and then flattens shows that gradient descent converged.

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
# scikit-learn's SVC with kernel="linear" minimizes the same objective, but
# solves it with a different algorithm (SMO, via the libsvm library).
# If our implementation is correct, both models should land on nearly the
# same w and b.
#
# Cosine similarity compares the direction of the two weight vectors:
# 1.0 means they point exactly the same way.

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
# PART 7: DOES THE DATA NEED A NONLINEAR BOUNDARY?
# =============================================================================
# A linear SVM can only draw a flat boundary. An RBF kernel lets the SVM draw
# a curved boundary. If RBF clearly beats linear, the data is nonlinear.
# If they score about the same, a linear SVM is enough.
#
# The RBF kernel has two settings, C and gamma (how far each sample's
# influence reaches). GridSearchCV tries every combination below and picks
# the best one using 5-fold cross-validation on the training set only:
# the training set is split into 5 parts, and each part takes a turn as a
# validation set while the model trains on the other 4.

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
# Malignant is treated as the "positive" class, since catching cancer is the
# point of the model.
#   Accuracy:  share of all test samples classified correctly
#   Precision: of the samples predicted malignant, how many truly were
#   Recall:    of the truly malignant samples, how many the model caught
#              (the most important metric here: a missed cancer is the
#              costliest mistake)
#   F1:        a single score that balances precision and recall
#
# The confusion matrix shows counts of correct and incorrect predictions
# for each class.

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
