# SVM Breast Cancer Classification

A linear Support Vector Machine written from scratch in NumPy, trained to
classify tumors in the Breast Cancer Wisconsin (Diagnostic) dataset as
malignant or benign. The model is checked against scikit-learn's SVC, and an
RBF-kernel SVC is used to test whether the data needs a nonlinear boundary.

## Files

| File | Purpose |
|---|---|
| `svm_breast_cancer.py` | All project code: the from-scratch SVM, preprocessing, training, comparison, and evaluation |
| `requirements.txt` | Library versions |

## Environment

- Python 3.13.18
- numpy 2.5.3
- scikit-learn 1.9.1
- matplotlib 3.11.2

The dataset is built into scikit-learn, so no download is needed.

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python svm_breast_cancer.py
```

The script runs in about 10 seconds. It prints results for each step to the
terminal, saves two plots to `figures/` (`loss_curve.png` and
`confusion_matrices.png`), and opens them in a window at the end.

## Steps in the script

1. Define the `LinearSVM` class (hinge loss, gradient descent)
2. Load 569 samples with 30 features; map labels to -1 (malignant) / +1 (benign)
3. Stratified 80/20 train/test split
4. Scale features with `StandardScaler`, fit on training data only
5. Train our `LinearSVM` and plot the loss curve
6. Train scikit-learn's linear SVC with the same C and compare weights and bias
7. Tune an RBF-kernel SVC with 5-fold cross-validation
8. Report test accuracy, precision, recall, F1, and confusion matrices
