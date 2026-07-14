import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

## scikit-learn Question 1
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])
new_years = np.array([8,10]).reshape(-1,1)

model = LinearRegression()
model.fit(years, salary)
salary_predic = model.predict(new_years)

print(f"Salary Predictions: {salary_predic}")
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

## scikit-learn Question 2
x = np.array([10, 20, 30, 40, 50])
print(f"X Shape: {x.shape}")
x = x.reshape(-1,1)
print(f"New Shape: {x.shape}")

# scikit expects two arguments for it to work. 
# In the case of a single array, reshaping it 
# tells scikit learn that with this data there 
# is only 1 column of information.

## scikit-learn Question 3
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
centers = kmeans.cluster_centers_
labels = kmeans.predict(X_clusters)

plt.scatter(X_clusters[:,0], X_clusters[:,1], c=kmeans.labels_, cmap='viridis', s=60, alpha=0.7)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centers')
plt.title('Data Clusters')
plt.xlabel('X')
plt.ylabel('Y')

plt.tight_layout()
plt.savefig('assignment_02/outputs/kmeans_clusters.png')
plt.show()

print(f"Centers: {kmeans.cluster_centers_}")
print(f"Cluster Count: {np.bincount(labels)}")

## Linear Regression Data
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

## Linear Regression Question 1
plt.figure(2)
plt.scatter(age,cost, c=smoker,cmap="coolwarm")
plt.title('Medical Cost vs Age')
plt.xlabel('Age')
plt.ylabel('Medical Cost')
plt.savefig('assignment_02/outputs/cost_vs_age.png')
plt.show()

# People who smoke consistantly have 
# higher medical costs comparded to those who don't.

## Linear Regression Question 2
age = age.reshape(-1,1)
age_train, age_test, cost_train, cost_test = train_test_split(
    age, cost, test_size=0.2, random_state = 42)

print(f"Age Training Shape:{age_train} ")
print(f"Age Testing Shape:{age_test} ")
print(f"Cost Training Shape:{cost_train} ")
print(f"Cost Training Shape:{cost_test} ")

## Linear Regression Question 3
model = LinearRegression()
model.fit(age_train, cost_train)

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

cost_predict = model.predict(age_test)
rmse = np.sqrt(mean_squared_error(cost_test, cost_predict))
r2 = model.score(age_test, cost_test)

print(f"Root Mean Squared Error: {rmse}")
print(f"R Squared: {r2}")

# The intercept indicates that medical costs 
# start at just over $7,000 dollars. The positive 
# slope shows a regular increase in medical costs 
# as age increases.

## Linear Regression Question 4
X_full = np.column_stack([age, smoker])
X_train, X_test, y_train, y_test = train_test_split(
    X_full, cost, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train, y_train)
r2 = model_full.score(X_test, y_test)
y_pred = model.predict(X_test)

print(f"R Squared: {r2}")
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])

# ADD COMMENT LATER

## Linear Regression Question 5
plt.figure(3)
plt.scatter(X_train, y_train, color = 'blue', label='Actual')
plt.plot(X_train, y_pred, color = 'red', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Cost')
plt.legend(labels)
plt.show()