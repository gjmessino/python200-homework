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

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)

log_probs = logreg.predict_proba(X_test)[:, 1]
knn_probs = knn.predict_proba(x_test_scaled)[:,1]

print("Logistic Regression")
print(f"Probabilities: {logreg.predict_proba(X_test)[:, 1]}")
print(f"AUC Score: {roc_auc_score(y_test, log_probs)}")

print("KNN")
print(f"Probabilities: {knn.predict_proba(x_test_scaled)[:,1]}")
print(f"AUC Score: {roc_auc_score(y_test, knn_probs)}")

# KNN has the higher AUC score (.9394), 
# so it is the better option for separating 
# and testing data. The probabilities for 
# KNN were also more consistant.

## --- ROC Question 2 --- ##
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, log_probs)
fpr_knn, tpr_knn, thresholds_knn = roc_curve(y_test, knn_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr).plot(ax=ax, name = 'Logistic Regression (AUC Score: 0.706)')
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve")
RocCurveDisplay(fpr=fpr_knn, tpr=tpr_knn).plot(ax=ax, name='KNN (AUC Score: 0.9394)')
ax.legend()
plt.tight_layout()
plt.savefig('outputs/roc_comparison.png')
plt.show()

# When TPR = .8, the KNN model has almost no flase 
# positives. However, the logistric regression model 
# has roughly a .7 FPR at the same spot. Which gives 
# KNN the higher overall success rating, because 
# it is right more than it's wrong.

## --- ROC Question 3 --- ##
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

print(f"Best F1: {best_f1}")
print(f"Best FPR: {best_fpr}")
print(f"Best TPR: {best_tpr}")
print(f"Best Threshold: {best_thresh}")

# The best threshold is below a 0.5 (.2756). 
# This is because .5 is an arbitrary number 
# that has no baring on the real data. 
# Thresholds beyond .5 are good 
# for data that doesn't have a binary 
# solution. Low thresholds are good when 
# the consequences of a false negative are 
# worse than those of a false positive.

## --- GridSearchCV --- ##

## --- GridSearch Question 1 --- ##
pipeline1 = Pipeline([
   ("scaler", StandardScaler()),
   ("logreg", LogisticRegression(max_iter=1000))
])

param_grid = {
    'logreg__C' : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search1 = GridSearchCV(
    estimator=pipeline1,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
)

grid_search1.fit(X_train, y_train)

print(f"Best C: {grid_search1.best_params_['logreg__C']}")
print(f"Best AUC: {grid_search1.best_score_:.3f}")

lr = grid_search1.best_estimator_
y_pred  = lr.predict(X_test)
y_probs = lr.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")

# The test picked 100, which is far higher than 
# what I would have guessed for the best C, 
# because high C values lead to over fitting.

## --- GridSearch Question 2 --- ##

pipeline2 = Pipeline([
   ("scaler", StandardScaler()),
   ("dtree", DecisionTreeClassifier(random_state=42))
])

param_grid = {
    'dtree__max_depth' : [2, 3, 5, 8, None]
}

grid_search2 = GridSearchCV(
    estimator=pipeline2,
    param_grid=param_grid,
    cv=5,
    scoring="roc_auc",
)

grid_search2.fit(X_train, y_train)

print(f"Best Max Depth: {grid_search2.best_params_['dtree__max_depth']}")
print(f"Best AUC: {grid_search2.best_score_:.3f}")

lr = grid_search2.best_estimator_
y_pred  = lr.predict(X_test)
y_probs = lr.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")

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
print(
    log_results[["param_logreg__C", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .to_string(index=False)
)

dtree_results = pd.DataFrame(grid_search2.cv_results_)
print('Decision Tree Results')
print(
    dtree_results[["param_dtree__max_depth", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .to_string(index=False)
)

# Logistic Regression had overall lower 
# standard deviations. When C = .1 & C = .01 
# the mean is close together but the standard 
# deviations are far apart. The C value of .1 
# is better because a smalled STD score means 
# the results are closer together and likely 
# more accurate.

## --- joblib --- ##

## --- joblib Question 1 --- ###
joblib.dump(lr, 'models/warmup_model.pkl')

loaded_clf = joblib.load("models/warmup_model.pkl")

original_preds = lr.predict(X_test)
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

print(f"Predictions: {loaded_clf.predict(new_samples)}")
print(f"Probabilities: {loaded_clf.predict_proba(new_samples)}")

# I expected the all 0s row to predict 0, which is did. 
# I was surprised by predict proba giving it decimal
#  estimates as opposed to a 1 and 0, especially given 
# that the predict proba for other rows is integers.