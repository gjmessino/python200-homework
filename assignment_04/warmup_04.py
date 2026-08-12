import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
    f1_score
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

## --- Part 1: Warmup Exercises --- ###

## --- ROC and AUC --- ##

## --- ROC Question 1 --- ##
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)

knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5))
])
knn_pipe.fit(X_train, y_train)

log_probs = logreg.predict_proba(X_test)[:, 1]
knn_probs = knn_pipe.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, log_probs)
knn_auc = roc_auc_score(y_test, knn_probs)

print("Logistic Regression")
print(f"AUC Score: {lr_auc}")

print("KNN")
print(f"AUC Score: {knn_auc}")

# Logistic Regression achieves a higher 
# AUC score (~0.92) compared to KNN (~0.88).
# This tells us that Logistic Regression 
# provides superior class separation overall 
# across all possible decision thresholds, 
# making it a stronger classifier for this 
# dataset regardless of the specific 
# threshold selected.

## --- ROC Question 2 --- ##
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, log_probs)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr).plot(ax=ax, name = f'Logistic Regression (AUC Score: {lr_auc})')
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve")
RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn).plot(ax=ax, name=f'KNN (AUC Score: {knn_auc})')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/roc_comparison.png')
plt.show()

# At TPR = 0.80 (80% True Positive Rate) 
# Logistic Regression has a lower FPR (~0.05)
# compared to KNN's FPR (~0.12). Practically, 
# if you need to catch 80% of actual positive 
# cases, Logistic Regression is the better 
# choice because it produces significantly 
# fewer false alarms (false positives).

## --- ROC Question 3 --- ##
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, log_probs)

best_f1 = 0
best_fpr = 0
best_tpr = 0
best_thresh = 0

for threshold, f, t in zip(thresholds_lr, fpr_lr, tpr_lr):
    y_pred = (log_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_fpr = f
        best_tpr = t
        best_thresh = threshold

print(f"Optimal Threshold: {best_thresh:.4f}")
print(f"TPR at Optimum: {best_tpr:.4f}")
print(f"FPR at Optimum: {best_fpr:.4f}")
print(f"Best F1 Score: {best_f1:.4f}")

# The optimal threshold found by maximizing 
# F1 score (~0.28 to 0.40 depending on seed) 
# is lower than the default cutoff of 0.5. 
# The default 0.5 threshold assumes equal cost 
# for False Positives and False Negatives, 
# which rarely reflects real-world needs. 
# You would choose a threshold lower than 
# 0.5 in applications where missing a positive 
# case (False Negative) is significantly 
# more costly than a false alarm (False Positive), 
# such as disease diagnosis, fraud detection, 
# or safety warnings.

## --- GridSearchCV --- ##

## --- GridSearch Question 1 --- ##
pipeline1 = Pipeline([
   ("scaler", StandardScaler()),
   ("logreg", LogisticRegression(max_iter=1000))
])

param_grid1 = {
    'logreg__C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search1 = GridSearchCV(
    estimator=pipeline1,
    param_grid=param_grid1,
    cv=5,
    scoring="roc_auc",
)

grid_search1.fit(X_train, y_train)

default_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(C=1.0, max_iter=1000, random_state=42))
])
default_pipe.fit(X_train, y_train)
default_test_auc = roc_auc_score(y_test, default_pipe.predict_proba(X_test)[:, 1])

best_lr = grid_search1.best_estimator_
best_test_auc = roc_auc_score(y_test, best_lr.predict_proba(X_test)[:, 1])

print(f"Best C value: {grid_search1.best_params_['logreg__C']}")
print(f"Best CV AUC score: {grid_search1.best_score_:.4f}")
print(f"Test AUC (Best C): {best_test_auc:.4f}")
print(f"Test AUC (C=1.0): {default_test_auc:.4f}")

# The grid search picked C = 0.1 (or 1.0 
# depending on solver precision) instead 
# of default C=1.0. The test AUC difference 
# between C=1.0 and the best C value is 
# minimal (~0.000 to 0.002), showing that
# Logistic Regression is quite robust to 
# C variation on this dataset. The test 
# picked 100, which is far higher than 
# what I would have guessed for the best C, 
# because high C values lead to over fitting.

## --- GridSearch Question 2 --- ##

pipeline2 = Pipeline([
   ("scaler", StandardScaler()),
   ("dtree", DecisionTreeClassifier(random_state=42))
])

param_grid2 = {
    'dtree__max_depth' : [2, 3, 5, 8, None]
}

grid_search2 = GridSearchCV(
    estimator=pipeline2,
    param_grid=param_grid2,
    cv=5,
    scoring="roc_auc",
)

grid_search2.fit(X_train, y_train)
best_dt = grid_search2.best_estimator_
dt_test_auc = roc_auc_score(y_test, best_dt.predict_proba(X_test)[:, 1])

print(f"Best Max Depth: {grid_search2.best_params_['dtree__max_depth']}")
print(f"Best AUC: {grid_search2.best_score_:.3f}")
print(f"Test AUC (Tree): {dt_test_auc:.4f}")

# The AUC for the Decision Tree is much 
# higher than for Logistic Regression.
# While AUC is important part of this 
# difference may be attributed to using 
# C versus max depth, given that C 
# focuses on avoiding miscaluculations 
# and max depth is about setting 
# restrictions on data.

## --- GridSearch Question 3 --- ##
log_results = pd.DataFrame(grid_search1.cv_results_)
print('Logistic Regression Results')
sorted_results = (
    log_results[["param_logreg__C", "mean_test_score", "std_test_score"]]
    .sort_values(by="mean_test_score", ascending=False)
)
print("\nLogistic Regression CV Results (Sorted Best to Worst):")
print(sorted_results.to_string(index=False))

# Logistic Regression had overall lower 
# standard deviations. When C = .1 & C = .01 
# the mean is close together but the standard 
# deviations are far apart. The C value of .1 
# is better because a smalled STD score means 
# the results are closer together and likely 
# more accurate.

## --- joblib --- ##

## --- joblib Question 1 --- ###
best_lr_pipe = grid_search1.best_estimator_
joblib.dump(best_lr_pipe, 'models/warmup_model.pkl')

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")
# If the data had not been scaled the log reg 
# model might have weighed everything differntly 
# leading to different optimal results that may 
# be inaccurate.

## --- joblib Question 2 --- ###

## --- Simulated prediction script --- ##
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

for i, sample in enumerate(new_samples, start=1):
    sample_reshaped = sample.reshape(1, -1)
    pred_class = loaded_clf.predict(sample_reshaped)[0]
    pred_prob = loaded_clf.predict_proba(sample_reshaped)[0][1] 
    
    print(f"Row {i}: Predicted Class = {pred_class}, Probability (Class 1) = {pred_prob:.4f}")

# I expected the all 0s row to predict 0, which is did. 
# I was surprised by predict proba giving it decimal
# estimates as opposed to a 1 and 0, especially given 
# that the predict proba for other rows is integers.