import joblib
import json
import pandas as pd
import sys

## --- Task 1: Load and Verify --- ##
try:
    clf = joblib.load('models/weather_classifier.pkl')
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
    "temperature_2m_max": [20, 38, -2, 26, 15],
    "temperature_2m_min": [12, 25, -10, 5, 3],
    "precipitation_sum":  [0.0, 0.0, 12.0, 2.5, 3.0],
    "wind_speed_10m_max": [10, 45, 20, 28, 29]
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
    print(f"Day {i+1}: {features_str}")
    print(f"Predicted: {pred_label} (confidence: {prob:.2%})")

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
# can't be swayed by predictions. If we 
# ran weather predict first the code would 
# not work given that it uses a model build 
# in train_weather_classifier, therefor there 
# would be no model to predict with.

# To accurately predict the next days weather 
# the model would need to be looking at weather 
# from the past couple days and/or the same date 
# in previous years, as opposed to looking at 
# a years worth of data. To truly be correct 
# we would also need to incorporate aspects of 
# meteorology and how that area of study 
# predicts weather.