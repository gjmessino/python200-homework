import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
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
print(f"X Train Mean: {x_train_scaled.mean}")

means = x_train_scaled.mean(axis=0)
for i in range(len(x_train.columns)):
    col_name = x_train.columns[i]
    mean_val = means[i]
    print(f"{col_name}: {mean_val:.4e}")

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
    print(f"CV: {scores.mean}")
# The best k value is 15 becuase it 
# consistantly had the highest accuracy 
# rate, and the fewest overfits.

# --- Classifier Evaluation Question 1 --- #
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
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print(f"Classicifation Report: {classification_report(y_test, y_pred)}")
# KNN and Decision Tree accuracy are mostly similar. 
# While the accuracy score is marginally higher 
# for the Decision Tree, the classification reports 
# are almost identical.

# Scaled/unscaled data should have a lesser (or non-existent)
#  effect on Decision Tree Data given that larger 
# numbers will not automatically be given more weight.

# --- Logistic Regression Question 1 --- #
c_vals = [0.01, 1, 100]
for c in c_vals:
    model = LogisticRegression(
        max_iter = 1000,
        solver = 'liblinear',
        C = c
        )
    ovr_model = OneVsRestClassifier(estimator=model)
    ovr_model.fit(x_train_scaled, y_train)
    print(ovr_model.estimators_)
    print(f"C Value: {c}")
    total = 0
    for est in ovr_model.estimators_:
        total += np.abs(est.coef_).sum()
    print(f"Total Coefficient: {total}")

# --- PCA --- #
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting


# --- PCA Question 1 --- #
print(f"Digit Shape: {X_digits.shape}")
print(f"Image Shape: {images.shape}")

fig, axes = plt.subplots(1, 10, figsize = (12,5))
for i in range(10):
    img_idx = np.where(y_digits == i)[0][0]
    ax = axes[i]
    ax.imshow(images[img_idx], cmap='gray_r')
    ax.set_title(f'Digit {i}')
plt.suptitle('Digit Display')
plt.tight_layout()
plt.savefig('assignment_03/outputs/sample_digits.png')
plt.show()
plt.close()

# --- PCA Question 2 --- #
pca = PCA(n_components=None)
pca.fit(X_digits)
scores = pca.transform(X_digits)

scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.savefig('assignment_03/outputs/pca_2d_projection.png')
plt.show()

#Similar digits are clustered in the same space,
# with some more spread out than others. For
# example, 9 digits is very compact, but 8 digits 
# is spread out.

# --- PCA Question 3 --- #
arr = np.cumsum(pca.explained_variance_ratio_)
n = np.argmax(arr >= 0.80) + 1

plt.plot(range(1, len(arr) +1), arr, marker = 'o')
plt.xlabel('Number of Comonents')
plt.ylabel('Cumulative Explained Varience')
plt.title('PCA Variance')
plt.axhline(y=0.80, color='r', linestyle='--', label='80% Threshold')
plt.legend()
plt.tight_layout()
plt.savefig('assignment_03/outputs/pca_variance_explained.png')
plt.show()
plt.close()

# You need roughly 15 components to reach above the 80% threshold.

# --- PCA Question 4 --- #
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_list = [2, 5, 15, 40]

fig, axes = plt.subplots(5,5, figsize=(10,10))

for i in range(5):
    ax = axes[0,i]
    ax.imshow(images[i], cmap='gray_r')
    ax.set_title(f'Image {i}')

for row, n in enumerate(n_list, start=1):
    for col in range(5):
        reconstructed_img = reconstruct_digit(col, scores, pca, n)
        ax = axes[row, col]
        ax.imshow(reconstructed_img, cmap='gray_r')
        ax.set_title(f"n = {n}")

plt.suptitle('PCA Reconstruction')
plt.tight_layout()
plt.savefig('assignment_03/outputs/pca_reconstructions.png')
plt.show()
plt.close()