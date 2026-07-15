import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing Question 1 --- #
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state = 42, stratify=y)
print(f"X Train Shape: {x_train.shape}")
print(f"X Test Shape: {x_test.shape}")
print(f"Y Train Shape: {y_train.shape}")
print(f"Y Test Shape: {y_test.shape}")

# --- Preprocessing Question 2 --- #
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(f"X Train Mean: {x_train_scaled}")
print(f"X Test Mean: {x_test_scaled}")

# If the scaler is fit to the test data 
# it means it's acciently "seen" it which 
# can influence results.

# --- KNN Question 1 --- #
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
knn_predict = knn.predict(x_test)

print(f"Accuracy Score: {accuracy_score(y_test, knn_predict)}")
print(f"Classification Report: {classification_report(y_test, knn_predict)}")

# --- KNN Question 2 --- #
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(x_train_scaled, y_train)
knn_predict = knn_scaled.predict(x_test_scaled)

print(f"Accuracy Score: {accuracy_score(y_test, knn_predict)}")
print(f"Classification Report: {classification_report(y_test, knn_predict)}")

# Scaling the data made it more realistics,
# going from 1.0 for everything to just
# under that, giving variation to items
# in the classification report.

# --- KNN Question 3 --- #
cv_scores = cross_val_score(knn, x_train, y_train, cv=5)
print(f"CV Score: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Standard Deviation: {cv_scores.std():.3f}")

# This result is more trustworthy because it's 
# looking at the same data in different ways 
# five times over, so mistakes that were made 
# during the first evaluation and missed are
# more likely to be cuaght on other folds.

# --- KNN Question 4 --- #
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=5)
    print(f"K: {k}")
    print(f"CV: {scores}")
# The best k value is 15 becuase it 
# consistantly had the highest accuracy 
# rate, and the fewest overfits.

# --- Classifier Evaluation Question 1 --- #
plt.figure(1)
cm = confusion_matrix(y_test, knn_predict)
disp = ConfusionMatrixDisplay(
    confusion_matrix = cm, 
    display_labels=iris.target_names)
disp.plot()
plt.title("KNN Confusion Matrix (Iris)")
plt.savefig('assignment_03/outputs/knn_confusion_matrix.png')
plt.show()

# The model confuses versicolor and virginica around 20% of the time.

# --- Decision Trees Question 1 --- #
