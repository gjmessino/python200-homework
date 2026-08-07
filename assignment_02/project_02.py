from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error

## Task 1: Load and Explore
df = pd.read_csv('assignment_02/student/student-mat.csv', sep=';')
print(f"Shape: {df.shape}")
print(f"Data Types: {df.dtypes}")
print(df.head())

plt.figure(1)
plt.hist(df['G3'], bins=21, color='purple')
plt.title('Distribution of Final Math Grades')
plt.xlabel('Grade')
plt.ylabel('Number of Students')
plt.savefig('outputs/g3_distribution.png')
plt.show()

## Task 2: Preprocess the Data
print(f"Old Shape: {df.shape}")
new_df = df.drop(df[df['G3'] == 0].index)
print(f"New Shape: {new_df.shape}")

# There were 395 rows, now it is down to 357 which 
# correlates with the histogram showing that just 
# under 40 people didn't show up to take the exam. 
# Keeping these rows in would have skewed the ability 
# to accurately predict test scores because those 
# students never took the test.

cols = ['schoolsup', 'internet', 'higher', 'activities']
new_df[cols] = new_df[cols].replace(['yes', 'no'], [1,0])
new_df['sex'] = new_df['sex'].replace(['F', 'M'], [0,1])

r, pval = stats.pearsonr(df['G3'], df['absences'])
r2, pval2 = stats.pearsonr(new_df['absences'], new_df['G3'])
print(f"Original Correlation: {r}")
print(f"New Correlation: {r2}")

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.scatter(df['G3'], df['absences'], color = 'blue', alpha=.7)
ax1.set_title('Original')
ax1.set_ylabel('Absences')
ax1.set_xlabel('Grades')

ax2.scatter(new_df['G3'], new_df['absences'], color='green', alpha=.7)
ax2.set_title('New')
ax2.set_ylabel('Absences')
ax2.set_xlabel('Grades')

plt.suptitle('Absences vs Grades')
plt.tight_layout()
plt.show()

# The later data shows there a negative correlation 
# between absences and grades (i.e. fewer 
# absences = high grades). But that is only true 
# for students who actually took the final test. The 
# absences of students who didn't take it make it look 
# like many students failed as opposed to not taking 
# the test at all.

## Task 3: Exploratory Data Analysis
pearsons = {}
numeric_features = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                     "absences", "freetime", "goout", "Walc"]
for feat in numeric_features:
    r, p = pearsonr(new_df[feat], new_df['G3'])
    pearsons[feat] = r          # each feature gets its own key

pearson_sort = dict(sorted(pearsons.items(), key=lambda item: item[1]))
for k, v in pearson_sort.items():   # .items() gives (key, value) pairs
    print(f"{k:12s}: {v:+.3f}")

# Failures had the lowest correlation ratee (-.294). 
# It was in the negative that the higher the 
# number of failures, the less likely students
# were to get a good grade on the final test. 
# The highest correlation value (.19) was the mothers 
# education, showing students with educated 
# mothers were the most likely to pass. The feature 
# closest to zero was least likely to affect grades, 
# which was freetime (-.022). 

plt.figure(2)
plt.bar(pearsons.keys(), pearsons.values())
plt.xlabel('Features')
plt.ylabel('Correlation Values')
plt.title('Correlation Rates')
plt.savefig('outputs/correlation_rates.png')

# For this plot at looked at the distribution 
# of correlation values. It essentially does 
# what the earlier part of task 3 requested 
# but in a visual way, so it is easier to 
# understand. The differences between positive
# and negative correlations become much more 
# noticable.

plt.figure(3)
plt.hist(new_df['failures'], color = 'green', alpha = .7, label='Failures')
plt.hist(new_df['Medu'], color = 'pink', alpha=.7, label='Motherds Education')
plt.title('Failures vs Mothers Education')
plt.legend()
plt.show()
plt.savefig('outputs/failures_medu.png')

# For my second plot I looked at the two features with 
# the highest and lowest correlation values 
# (features vs medu). The graph shows how as one 
# decreases the other increases, demonstrating the 
# opposing effects of negative and positive correlations.

## Task 4: Baseline Model
fail = new_df['failures'].values.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(
    fail, new_df['G3'], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(x_test, y_test)

print(f"Slope: {model.coef_[0]}")
print(f"RMSE: {rmse}")
print(f"R Squared: {r2}")

# The negative slope shows that the more 
# failures there are, the lower the 
# grade is. RMSE indicates that the model's 
# grade predictions are off by just over 2.5 
# (or around %12.5). And R2 shows the model 
# is not doing a great job of predicting grades.

## Task 5: Build the Full Model
feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
X = new_df[feature_cols].values
y = new_df["G3"].values

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
train_r2 = model.score(x_train, y_train)
test_r2 = model.score(x_test, y_test)

print(f"RMSE: {rmse}")
print(f"Training R2: {train_r2}")
print(f"Testing R2: {test_r2}")

# Both testing and training R2s are much higher 
# than in question 4, showing the model is doing a 
# much better job at making predictions. Adding
# more features gives the model more information 
# about things that impact grades.

for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# schoolsup was the largest determinant of grade, 
# and I was surprised that it was negative, given 
# it seems like support would help students get a 
# higher grade. It could potential be because there 
# are only binary responses.
#
# In a real scenario I would drop free time, activities, 
# and abseces because they are the least likely to impact grades.

## Task 6: Evaluate and Summarize
plt.figure(4)
plt.scatter(y_pred,y_test, color = 'green')
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Predictions')
plt.ylabel('Real Data')
plt.title('Predicted vs Actual (Full Model)')
plt.savefig('outputs/predicted_vs_actual.png')
plt.show()

# The graph shows the model could use some 
# imrpovements given that there is a notable 
# cluster of dots, but the prediction line is 
# slightly off. The model is slightly more 
# off at the high end than the low end but 
# both extremes are the problem. The middle of 
# the data is fairly accurate.

feature_cols_g1 = feature_cols + ['G1']
X_g1 = new_df[feature_cols_g1].values

x_train, x_test, y_train, y_test = train_test_split(X_g1, y, test_size=0.2, random_state=42)
model_g1 = LinearRegression()
model_g1.fit(x_train, y_train)
y_pred = model_g1.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model_g1.score(x_test, y_test)

print(f"RMSE: {rmse}")
print(f"R Squared (with G1): {r2}")

# A high R^2 here does not mean G1 causes G3 -- it means G1 is a strong proxy for
# whatever is already driving a student's performance (ability, effort, home
# situation, etc.). G1 and G3 are both outcomes of the same underlying factors,
# measured at different points in the same course, so of course they move together.
#
# This makes the model much less useful for early intervention than the jump in R^2
# suggests. By the time G1 exists, a third of the term has already happened -- any
# student already struggling has already lost time. If educators want to flag
# at-risk students *before* G1 is available, they'd have to rely on the weaker but
# earlier-available background/behavioral features from the Task 5 model (failures,
# absences, study habits, parental education, etc.), even though that model explains
# less of the variance. The tradeoff is: wait for G1 and get a much more accurate
# but much later warning, or act early on a noisier signal.