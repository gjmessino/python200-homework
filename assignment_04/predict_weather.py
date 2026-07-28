import joblib
import json
import pandas as pd
import sys

## --- Task 1: Load and Verify --- ##
clf = joblib.load('models/weather_classifier.pkl')
try:
    with open('models/weather_classifier_metadata.json', 'r') as file:
        data = json.load(file)
except FileNotFoundError as e:
    print("Error: Missing model or metadata file!")
    print("Please run 'python train_weather_classifier.py' first to generate 'models/weather_classifier.pkl'.")
    sys.exit(1)

print(f"City:{data.get('city', 'Unknown')} (Lat: {data.get('latitude')}, Lon: {data.get('longitude')})")
print(f"Features: {data['features']}")
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

for i, row in new_days.iterrows():
    pred_label = "good" if predict[i] == 1 else "skip"
    prob = probs[i]

    features_str = (
        f"Max Temp: {row['temperature_2m_max']}°C, "
        f"Min Temp: {row['temperature_2m_min']}°C, "
        f"Precip: {row['precipitation_sum']}mm, "
        f"Wind: {row['wind_speed_10m_max']}km/h"
    )

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