import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ## Pandas Question 1
# data = {
#     "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
#     "grade":  [85, 72, 90, 68, 95],
#     "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
#     "passed": [True, True, True, False, True]
# }
# df = pd.DataFrame(data)

# print(f"First Three Rows: {df.head(3)}")
# print(f"Shape: {df.shape}")
# print(f"Data Types: {df.info()}")

# ## Pandas Question 2
# new_df = df[(df["grade"]>80) & (df["passed"] == True)]
# print (new_df)

# ## Pandas Question 3
# df["grade_curved"] = df["grade"] + 5
# print(df)

# ## Pandas Question 4
# df["name_upper"] = df["name"].str.upper()
# print(f"{df['name']}, {df['name_upper']}")

# ## Pandas Question 5
# df_grouped =  df.groupby("city").agg({'grade' : 'mean'})
# print(df_grouped)

# ## Pandas Question 6
# df['city'] = df['city'].replace('Austin', 'Houston')
# print(df['name'],df['city'])

# ## Pandas Question 7
# df = df.sort_values(by = 'grade', ascending = False)
# print(df.head(3))

# ## NumPy Question 1
# numarr = np.array([10, 20, 30, 40, 50])
# print(f"Shape: {numarr.shape}")
# print(f"dtype: {numarr.dtype}")
# print(f"ndim: {numarr.ndim}")

# ## NumPy Question 2
# arr = np.array([[1, 2, 3],
#                 [4, 5, 6],
#                 [7, 8, 9]])
# print(f"Shape: {arr.shape}")
# print(f"Size: {arr.size}")

# ## NumPy Question 3
# print(arr[0:2, 0:2])

# ## NumPy Question 4
# zeros = np.zeros((3,4))
# ones = np.ones((2,5))
# print(zeros)
# print(ones)

# ## NumPy Question 5
# arr = np.arange(0, 50, 5)
# print(arr)
# print(f"Shape: {arr.shape}")
# print(f"Mean: {arr.mean()}")
# print(f"Sum: {arr.sum()}")
# print(f"STD: {arr.std()}")

# ## NumPy Question 6
# arr = np.random.normal(scale=1, size=200)
# print(f"Mean: {arr.mean()}")
# print(f"STD: {arr.std()}")

## Matplotlib Question 1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure(1)
plt.plot(x,y)
plt.title("Squares")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

## Matplotlib Question 2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

plt.figure(2)
plt.bar(subjects,scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

## Matplotlib Question 3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure(3)
plt.scatter(x1,y1, color = 'purple', label = 'First Plot')
plt.scatter(x2,y2, color = 'pink', label = "Second Plot")
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Two Plots')
plt.legend()
plt.show()

## Matplotlib Question 4
plt.figure(4)
plt.subplot(x,y)
plt.subplot(subjects,scores)
plt.tight_layout()
plt.show()

##### COME BACK TO THIS ########

## Descriptive Stats Question 1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print(data.mean())
print(np.median(data))
print(np.var(data))
print(data.std())

## Descriptive Stats Question 2
data = np.random.normal(65, 10, 500)

plt.figure(5)
plt.hist(data, bins=20)
plt.title("Distribution of Scores")
plt.xlabel("Distribution")
plt.ylabel("Score Total")

## Descriptive Stats Question 3
