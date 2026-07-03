import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
import seaborn as sns

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
# x = [0, 1, 2, 3, 4, 5]
# y = [0, 1, 4, 9, 16, 25]

# plt.figure(1)
# plt.plot(x,y)
# plt.title("Squares")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# ## Matplotlib Question 2
# subjects = ["Math", "Science", "English", "History"]
# scores   = [88, 92, 75, 83]

# plt.figure(2)
# plt.bar(subjects,scores)
# plt.title("Subject Scores")
# plt.xlabel("Subjects")
# plt.ylabel("Scores")
# plt.show()

# ## Matplotlib Question 3
# x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
# x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

# plt.figure(3)
# plt.scatter(x1,y1, color = 'purple', label = 'First Plot')
# plt.scatter(x2,y2, color = 'pink', label = "Second Plot")
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Two Plots')
# plt.legend()
# plt.show()

# ## Matplotlib Question 4
# # plt.figure(4)
# # plt.subplot(x,y)
# # plt.subplot(subjects,scores)
# # plt.tight_layout()
# # plt.show()

# ##### COME BACK TO THIS ########

# ## Descriptive Stats Question 1
# data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

# print(np.mean(data))
# print(np.median(data))
# print(np.var(data))
# print(np.std(data))

# ## Descriptive Stats Question 2
# data = np.random.normal(65, 10, 500)

# plt.figure(5)
# plt.hist(data, bins=20)
# plt.title("Distribution of Scores")
# plt.xlabel("Distribution")
# plt.ylabel("Score Total")
# plt.show()

# ## Descriptive Stats Question 3
# group_a = [55, 60, 63, 70, 68, 62, 58, 65]
# group_b = [75, 80, 78, 90, 85, 79, 82, 88]

# plt.figure(6)
# plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
# plt.title("Score Comparison")
# plt.show()

# ## Descriptive Stats Question 4
# normal_data = np.random.normal(50, 5, 200)
# skewed_data = np.random.exponential(10, 200)

# plt.figure(7)
# plt.boxplot(normal_data, label="Normal")
# plt.boxplot(skewed_data, label = "Exponential")
# plt.title("Distribution Comparison")
# plt.show()

# ## ADD COMMENTS

# ## Descriptive Stats Question 5
# data1 = [10, 12, 12, 16, 18]
# print(f"Mean: {np.mean(data1)}")
# print(f"Median:{np.median(data1)}")
# print(f"Mode: {np.mode(data1)}")

# data2 = [10, 12, 12, 16, 150]
# print(f"Mean: {np.mean(data2)}")
# print(f"Median:{np.median(data2)}")
# print(f"Mode: {np.mode(data2)}")

## ADD COMMENT

## Hypothesis Question 1
# group_a = [72, 68, 75, 70, 69, 73, 71, 74]
# group_b = [80, 85, 78, 83, 82, 86, 79, 84]

# t_stat,p_val = stats.ttest_ind(group_a,group_b)
# print(f"Statistics: {t_stat}")
# print(f"P Value: {p_val}")

# ## Hypothesis Question 2
# alpha = 0.05
# if p_val < alpha:
#     print("Significant")
# else:
#     print("Insignificant")

# ## Hypothesis Question 3
# before = [60, 65, 70, 58, 62, 67, 63, 66]
# after  = [68, 70, 76, 65, 69, 72, 70, 71]

# t_stat, p_val = stats.ttest_rel(before, after)
# print(f"Statistics: {t_stat}")
# print(f"P Value: {p_val}")

# ## Hypothesis Question 4
# scores = [72, 68, 75, 70, 69, 74, 71, 73]
# t_stat, p_val = stats.ttest_1samp(scores, popmean = 70)
# print(f"Statistics: {t_stat}")
# print(f"P Value: {p_val}")

# ## Hypothesis Question 5
# t_stat, p_val = stats.ttest_ind(group_a, group_b, alternative = "less")
# print(f"P Value: {p_val}")

## Hypothesis Question 6
##ADD COMMENT

# ## Correlation Question 1
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]

# matrix = np.corrcoef(x, y)
# print(matrix)
# ## ADD COMMENT

# ## Correlation Question 2
# x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
# y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

# corr, p_val = pearsonr(x,y)
# print(f"Correlation: {corr}")
# print(f"P Value: {p_val}")

# ## Correlation Question 3
# people = {
#     "height": [160, 165, 170, 175, 180],
#     "weight": [55,  60,  65,  72,  80],
#     "age":    [25,  30,  22,  35,  28]
# }
# df = pd.DataFrame(people)
# matrix = df.corr()
# print(matrix)

# ## Correlation Question 4
# x = [10, 20, 30, 40, 50]
# y = [90, 75, 60, 45, 30]

# plt.figure(8)
# plt.scatter(x,y)
# plt.title("Negative Correlation")
# plt.xlabel("X Values")
# plt.ylabel("Y Values")
# plt.show()

# ## Correlation Question 5
# plt.figure(9)
# sns.heatmap(matrix, annot=True)
# plt.title("Correlation Heatmap")
# plt.show()

## Pipeline Question 1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def data_pipeline(arr):
    series = create_series(arr)
    clean = clean_data(series)
    my_dict = summarize_data(clean)
    return(my_dict)
def create_series(arr):
    return pd.Series(arr)
def clean_data(series):
    return series.dropna()
def summarize_data(series):
    my_dict = {'mean': series.mean(),
               'median': series.median(),
               'mode': series.mode()[0],
               'std': series.std()}
    return my_dict

my_dict = data_pipeline(arr)
print(f"Keys: {my_dict.keys()}")
print(f"Values: {my_dict.values()}")