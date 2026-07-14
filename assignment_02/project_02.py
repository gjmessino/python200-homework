from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
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
plt.savefig('assignment_02/outputs/g3_distribution.png')
plt.show()

## Task 2: Preprocess the Data
new_df = df.drop(df[df['G3'] == 0].index)
print(f"Shape: {new_df.shape}")

# There were 395 rows, now it is down to 357 which 
# correlates with the histogram showing that just 
# under 40 people didn't show up to take the exam. 
# Keeping these rows in would have skewed the ability 
# to accurately predict test scores because those 
# students never took the test.

cols = ['schoolsup', 'internet', 'higher', 'activities']
new_df[cols] = new_df[cols].replace(['yes', 'no'], [1,0])
new_df['sex'] = new_df['sex'].replace(['F', 'M'], [1,0])

r, pval = stats.pearsonr(df['G3'], df['absences'])
r2, pval2 = stats.pearsonr(new_df['absences'], new_df['G3'])
print(f"Original Correlation: {r}")
print(f"New Correlation: {r2}")

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(12,5))
ax1.scatter(df['G3'], df['absences'], color = 'blue', alpha=.7)
ax1.set_title('Original')
ax1.set_ylabel('Absences')
ax1.set_xlabel('Grades')

ax2.scatter(df['G3'], df['absences'], color='green', alpha=.7)
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
plt.figure(2)
plt.scatter(y,X, color = 'green')
plt.plot(y_pred, y, color='purple')
plt.xlabel('Predictions')
plt.ylabel('Real Data')
plt.title('Predicted vs Actual (Full Model)')
plt.savefig('assignment_02/outputs/predicted_vs_actual.png')
plt.show()

#ADD COMMENT

feature_cols.append('G1')
x_train, y_train, x_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = model.score(x_test, y_test)

print(f"RMSE: {rmse}")
print(f"R Squared: {train_r2}")

#ADD COMMENT