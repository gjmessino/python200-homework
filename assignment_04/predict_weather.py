import joblib
import json
import pandas as pd

## --- Task 1: Load and Verify --- ##
clf = joblib.load('models/weather_classifier.pkl')
with open('models/weather_classifier_metadata.json', 'r') as file:
    data = json.load(file)
print(f"Features: {data['features']}")
print(f"City: {data['trained_on']}")
print(f"Test AUC: {data['test_auc']}")

## --- Task 2: Predict on New Data --- ##
new_days = pd.DataFrame({
    "temperature_2m_max": [10, 50, 0, 15, 7], 
    "temperature_2m_min": [10, 20, -5, 4, 0],
    "precipitation_sum":  [1, 5, -8, 3, 1],
    "wind_speed_10m_max": [2, 40, 25, 29, 29]
})

predict = clf.predict(new_days)
probs = clf.predict_proba(new_days)[:,1]

for i, (pred, prob) in enumerate(zip(predict, probs)):
    label = "good" if pred == 1 else "skip"
    print(f"Day {i+1}: {label} ({prob:.2f} probability)")

## --- Task 3: Reflect --- ##

# For day 5 the probability came out to 51%, 
# in my area specifically, I'd say that's a 
# good running day. However, probability 
# around 50% in other circumstances might 
# need to be handled case by case, or we 
# may need another variable such as what 
# the tempuratue feels like.

# Keeping the model in a separate file 
# makes it easier to use in multiple 
# circumstances. It also keeps the model 
# acccurate to the training data so it 
# can't be swayed by predictions.

# To accurately predict the next days weather 
# the model would need to be looking at weather 
# from the past couple days and/or the same date 
# in previous years, as opposed to looking at 
# a years worth of data. To truly be correct 
# we would also need to incorporate aspects of 
# meteorology and how that area of study 
# predicts weather.