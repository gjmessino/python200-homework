import joblib
import json
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import requests
import sklearn
import sys

## --- Step 1: Fetch the Data --- ##
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}
response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)
df.info()

## --- Step 2: Engineer Labels --- ##
def label_running_day(row):
    """Return 1 if conditions are good for an outdoor run, 0 otherwise."""
    temp_ok    = 7 <= row["temperature_2m_max"] <= 26   # 45–84°F 
    above_freeze = row["temperature_2m_min"] >= 0        # above freezing at dawn
    dry        = row["precipitation_sum"] < 3.0          # light rain or less
    not_windy  = row["wind_speed_10m_max"] < 30          # under 30 km/h
    return int(temp_ok and above_freeze and dry and not_windy)

df["good_for_running"] = df.apply(label_running_day, axis=1)

print(df["good_for_running"].value_counts())
print(f"\nFraction of good days: {df['good_for_running'].mean():.2f}")
print(df.describe())
# 61% of the days here (San Francisco) are good for 
# running. It's a little lower than I expected given 
# that it's never too hot or too cold, and rain is 
# about the only weather that stops people from going outside.

## --- Step 3: Train and Tune --- ##
FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000))
])

param_grid = {
    'clf__C' : [.01, .1, .5, 1, 5, 10]
}

grid_search = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv = 5,
    scoring="roc_auc"
)

grid_search.fit(X_train, y_train)
print(f"Best C: {grid_search.best_params_['clf__C']}")
print(f"Best AUC: {grid_search.best_score_:.3f}")

best_lr = grid_search.best_estimator_
y_preds = best_lr.predict(X_test)
y_probs = best_lr.predict_proba(X_test)[:,1]
test_auc = roc_auc_score(y_test, y_probs)
print("Classification Report")
print(classification_report(y_test, y_preds))
print(f"Test AUC: {test_auc}")

fpr, tpr, thresholds = roc_curve(y_test, y_probs)

fig, ax = plt.subplots(figsize=(6,5))
RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax, name='Logistic Regression')
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve — Weather Classifier")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/weather_roc.png")
plt.show()

## --- Step 4: Reflect on Evaluation --- ##

# The AUC is very accurate (96%). It better than 
# I expected given the probability of it being 
# a good running day is close-ish to 50/50, so 
# random guessing could theoretically work. But 
# here clearly the model is better.

# Precision and recall are identical in every 
# category meaning there are the same amount 
# of False positives and false negatives. 
# Theoretically there should be slightly more 
# false positives because we know there are 
# more positive answers overall.

# I would not use a default of .5, because 
# the model is much more successful and random 
# guessing.

## --- Step 5: Save the Model --- ##
os.makedirs("models", exist_ok=True)
joblib.dump(best_lr, 'models/weather_classifier.pkl')

metadata = {
    "python_version": sys.version,
    "sklearn_version": sklearn.__version__,
    "features": FEATURES,
    "best_params": grid_search.best_params_,
    "test_auc": round(test_auc, 4),
    "city": "San Francisco, CA",
    "latitude": 37.7749,
    "longitude": -122.4194,
    "label_thresholds": {
        "temperature_2m_max": "7–26°C",
        "temperature_2m_min": ">= 0°C",
        "precipitation_sum":  "< 3.0 mm",
        "wind_speed_10m_max": "< 30 km/h",
        # Comment: Used the same label thresholds as above for consistency.
    },
    "label_description": (
        "A day is labeled 'good for running' when the high temperature is between "
        "7-26°C (45-79°F), the low temperature stays at or above freezing (0°C), "
        "total precipitation is under 3.0 mm, and max wind speed is under 30 km/h. "
        "All four conditions must hold."
    ),
}

with open('models/weather_classifier_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("Model and metadata saved to models/")
