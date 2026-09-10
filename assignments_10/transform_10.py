from dotenv import load_dotenv
import joblib
import json
from openai import OpenAI
import os
import pandas as pd
from supabase import create_client

## Step 1: Incremental Read ##
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
response = supabase.table("weather_raw").select("*").execute()
raw_rows = response.data
print(f"Fetched {len(raw_rows)} rows from weather_raw")

enriched_response = supabase.table("weather_enriched").select("date").execute()
already_done = {row["date"] for row in enriched_response.data}
to_classify = [row for row in raw_rows if row["date"] not in already_done]
print(f"Records to classify: {len(to_classify)} (skipping {len(already_done)} already enriched)")

## Step 2: ML Transform ##
with open("models/weather_classifier_metadata.json") as f:
    metadata = json.load(f)
FEATURES = metadata["features"]
df = pd.DataFrame(to_classify)
X = df[FEATURES]

clf = joblib.load("models/weather_classifier.pkl")
predictions  = clf.predict(X)
probabilities = clf.predict_proba(X)[:, 1]

print(f"Good days predicted: {predictions.sum()} / {len(predictions)}")
print(f"Confidence range: {probabilities.min():.2f} – {probabilities.max():.2f}")

enrichment_records = []
for i, row in enumerate(to_classify):
    enrichment_records.append({
        "date": row["date"],
        "good_for_running": bool(predictions[i]),
        "confidence": round(float(probabilities[i]), 4),
    })

print("Sample enrichment records:")
for r in enrichment_records[:3]:
    print(r)

## Step 3: LLM Transform ##
SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

def make_user_message(row, good_for_running, confidence):
    prediction_text = "good for running" if good_for_running else "not ideal for running"
    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} (confidence: {confidence:.0%})"
    )

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

for i, record in enumerate(enrichment_records):
    try:
        raw_row = next(r for r in to_classify if r["date"] == record["date"])

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": make_user_message(
                        raw_row,
                        record["good_for_running"],
                        record["confidence"],
                    ),
                },
            ],
            max_tokens=100,
        )

        summary = response.choices[0].message.content.strip()
        record["llm_summary"] = summary
    except:
        print('An error has occured')
        continue

    if (i + 1) % 50 == 0:
        print(f"LLM Enriched {i + 1} / {len(enrichment_records)} records...")

## Step 4: Load ##
response = (
    supabase.table("weather_enriched")
    .upsert(enrichment_records, on_conflict="date")
    .execute()
)
print(f"Upserted {len(response.data)} rows into weather_enriched")

## Step 5: Verify ##
response = supabase.table("weather_enriched").select("*").execute()
print(f"Total Number of Rows: {len(response.data)}")
print(f"Sample Rows...")
sample = supabase.table("weather_enriched").select("*").limit(5).execute()
for sam in sample:
    print(f"Date: {sam['date']}")
    print(f"Good for Running: {sam['good_for_running']}")
    print(f"Confidence: {sam['confidence']}")
    print(f"LLM Response: {sam['llm_response']}")

## Step 6: Reflect ##
# 1. In this hypothetical, if the model is trained to have the same ratings for good running days (ex same max/min 
# tempuratures), then the location change wouldn't matter. If the standards for a good running day are the same 
# everywhere then predicting conditions in one location won't change anything. However, the week 4 assignment allowed 
# us to adjust our standards for what makes a good running day if we change locations (ex. here is SF people are okay 
# with running on hotter days because it is rarely warm here). If the code changes the standard for good_for_running 
# based on location then we would end up with incorrect predictions based on location. 

# 2. It is the LLMs job to interpret the data from the ML into understandable language. Because it is not outputing 
# the exact data to the user it's possible for its interpretation to skew how the data is presented. 

# 3. Latency would be one of the largest concern given that there is a call to the LLM every time a record is processed. 
# This takes far longer than simply printing the predictions as classified by the ML. To fix this we could only do LLM 
# calls on dates requested by a user, so instead of getting a written response for every new row, we'd only make the 
# LLM response if necessary. 