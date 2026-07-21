import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Task 1: Load and Explore --- #
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.head())

cats = ['word_freq_free', 'char_freq_!', 'capital_run_length_total']
for label in cats:
    spam = df.loc[df['spam_label'] == 1, label]
    ham = df.loc[df['spam_label'] == 0, label]
    plt.boxplot([spam,ham], labels = ["Spam", "Not Spam"])
    plt.title(label)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(f'assignment_03/outputs/{label}.png')
    plt.show()

# --- Task 2: Prepare Your Data --- #
x = df.drop('spam_label', axis=1)
y = df['spam_label']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state = 42, stratify=y
    )

# --- PCA preprocessing --- #
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

pca = PCA()
pca.fit(x_train_scaled)

arr = np.cumsum(pca.explained_variance_ratio_)
n = np.argmax(arr >= 0.90) + 1

plt.plot(range(1, len(arr) +1), arr, marker = 'o')
plt.xlabel('Number of Comonents')
plt.ylabel('Cumulative Explained Varience')
plt.title('PCA Variance')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('assignment_03/outputs/spam_pca.png')
plt.show()
plt.close()

X_train_pca = pca.transform(x_train_scaled)[:, :n]
X_test_pca  = pca.transform(x_test_scaled)[:, :n]

# --- Task 3: A Classifier Comparison --- #
unscaled_knn = KNeighborsClassifier(n_neighbors=5)
unscaled_knn.fit(x_train, y_train)
knn_predict1 = unscaled_knn.predict(x_test)
print(f"Unscaled Accuracy Score (KNN): {accuracy_score(y_test, knn_predict1)}")
print(f"Classification Report: {classification_report(y_test, knn_predict1)}")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)
knn_predict2 = knn.predict(x_test_scaled)
print(f"Scaled Accuracy Score(KNN): {accuracy_score(y_test, knn_predict2)}")
print(f"Classification Report: {classification_report(y_test, knn_predict2)}")

depth = [3, 5, 10, None]
for d in depth: 
    dtree = DecisionTreeClassifier(max_depth = d, random_state = 42)
    dtree.fit(x_train, y_train)

    train_pred = dtree.predict(x_train)
    test_pred = dtree.predict(x_test)

    print(f"Max Depth: {d}")
    print(f"Train Accuracy Score(Decision Tree): {accuracy_score(y_train, train_pred)}")
    print(f"Test Accuracy Score(Decision Tree): {accuracy_score(y_test, test_pred)}")

    print(f"Feature Importance: {dtree.feature_importances_}")


    # Accuracy increases with max depth because there 
    # is more data to work with. I would pick none as 
    # the max depth given that for both the test and 
    # train data it had the highest accuracy.

dtree = DecisionTreeClassifier(max_depth = None, random_state = 42)
dtree.fit(x_train, y_train)
dtree_predict = dtree.predict(x_test)
print(f"Scaled Accuracy Score(KNN): {accuracy_score(y_test, dtree_predict)}")
print(f"Classification Report: {classification_report(y_test, dtree_predict)}")

importance_df_tree = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': dtree.feature_importances_})

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
print(f"Important Features: {rf.feature_importances_}")
importance_df_rf = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': rf.feature_importances_})

logreg_scale = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
logreg_scale.fit(x_train_scaled, y_train)
logreg_scale_pred = logreg_scale.predict(x_test_scaled)
print(f"Accuracy (Scaled): {accuracy_score(y_test, logreg_scale_pred):.4f}")
print(f"Classification Report (Scaled): {classification_report(y_test, logreg_scale_pred)}")

logreg_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
logreg_pca.fit(X_train_pca, y_train)
logreg_pca_pred = logreg_pca.predict(X_test_pca)
print(f"Accuracy (PCA): {accuracy_score(y_test, logreg_pca_pred):.4f}")
print(f"Classification Report (PCA): {classification_report(y_test, logreg_pca_pred)}")
# The scaled data had a marginally higher accuracy score (.9294 vs 0.9186 for PCA).

# Unscaled KNN data had the lowest 
# accuracy overall, problem because 
# it was weighting certain data heavier 
# than other. The Decision tree with a max 
# iter of Non had an almost perfect 
# accuracy score. Other testing fell in 
# the middle with accuracy mostly in the 
# low 90th percentile.

top10_rf = importance_df_rf.nlargest(10, 'Importance')
top10_tree = importance_df_tree.nlargest(10, 'Importance')

fig, [ax1, ax2] = plt.subplots(1,2, figsize=(14, 6))
ax1.bar(top10_rf['Feature'], top10_rf['Importance'])
ax1.set_title('Important Features (Random Forest)')
ax1.set_xlabel('Features')

ax2.bar(top10_tree['Feature'], top10_tree['Importance'])
ax2.set_title('Important Features (Decision Tree)')
ax2.set_xlabel('Features')

plt.suptitle('Feature Important')
plt.tight_layout()
plt.savefig('assignment_03/outputs/feature_importances.png')
plt.show()

# --- Task 4: Cross-Validation --- #
def get_cv(model, x, y, label):
    cv_scores = cross_val_score(model, x, y, cv=5)
    print(label)
    print(f"Cross Validation Score: {cv_scores}")
    print(f"Mean: {cv_scores.mean():.3f}")
    print(f"Standard Deviation: {cv_scores.std():.3f}")

get_cv(unscaled_knn, x_train, y_train, 'Unscaled KNN')
get_cv(knn, x_train_scaled, y_train, 'Scaled KNN')
get_cv(knn, X_train_pca, y_train, 'PCA KNN')
get_cv(dtree, x_train, y_train, 'Decision Tree')
get_cv(rf, x_train, y_train, 'Random Forest')
get_cv(logreg_scale, x_train_scaled, y_train, 'Scaled Logistical Regression')
get_cv(logreg_pca, x_train_scaled, y_train, 'Scaled Logistical Regression')

# --- Task 5: Building a Prediction Pipeline --- #
non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),  # 'n' calculated in Task 2
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

non_tree_pipeline.fit(x_train, y_train)
non_tree_pred = non_tree_pipeline.predict(x_test)
print(f"Classification Report (Non Tree): {classification_report(y_test, non_tree_pred)}")

tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
tree_pipeline.fit(x_train, y_train)
tree_pred = tree_pipeline.predict(x_test)

print(f"Classification Report (Tree): {classification_report(y_test, tree_pred)}")

# The non-tree pipeline includes preprocessing steps (StandardScaler and PCA) 
# because distance- and gradient-based algorithms (like Logistic Regression and KNN) 
# are sensitive to differing feature scales and high dimensionality. The tree pipeline 
# contains only the classifier because Decision Trees and Random Forests split features 
# independently on single feature thresholds and are invariant to scaling.
#
# Packaging models into Pipelines prevents data leakage (scalers/PCA fit strictly on 
# training folds during cross-validation), simplifies deployment, and ensures that raw 
# unseen test data undergoes identical transformations without manual bookkeeping.