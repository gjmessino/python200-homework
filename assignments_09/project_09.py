import os
import datetime
from dotenv import load_dotenv
import requests
from supabase import create_client
import datetime

## Step 1: Extract ##
url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude":  37.774,
    "longitude": -122.419,
    "start_date": "2023-01-01",
    "end_date":   "2023-12-31",
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
data = response.json()

## Step 2: Transform ##
daily = data["daily"]
records = [
    {
        "date":               daily["time"][i],
        "temperature_2m_max": daily["temperature_2m_max"][i],
        "temperature_2m_min": daily["temperature_2m_min"][i],
        "precipitation_sum":  daily["precipitation_sum"][i],
        "wind_speed_10m_max": daily["wind_speed_10m_max"][i],
    }
    for i in range(len(daily["time"]))
]

print(f"Prepared {len(records)} records")
print("First record:", records[0])
print("Last record:", records[-1])

# Comment
# I have 365 records which is the exact number I expected. 1 record for every day of the year.

## Step 3: Load ##
load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

response = supabase.table("weather_raw").upsert(records, count="exact").execute()
print(f"\nRows Upserted: {response.count}")

response = supabase.table("weather_raw").upsert(records, count="exact").execute()
print(f"Second attempt Rows Upserted: {response.count}")

# Comment
# The number of rows affected is 365 in both instances. This means that with idempotency the record 
# is constantly being updated. Even though the new data didn't change any of the existing data, 
# Supabase still updated each column by key.

## Step 4: Verify ##
response = supabase.table("weather_raw").select("*", count="exact").execute()
print(f"\nTotal Number of Rows: {response.count}") 
print(f"Earliest Date: {response.data[0]}")
print(f"Latest Date: {response.data[-1]}")

julyFourth = supabase.table("weather_raw").select("*").eq("date", "2023-07-04").execute()
if len(julyFourth.data) > 0:
    print(f"July 4th Record: {julyFourth}")

else:
    past = (supabase
            .table('weather_raw')
            .select('*')
            .lte('date', julyFourth)
            .order("date_column", desc=True)
            .limit(1)
            .execute())
    future = (supabase
            .table('weather_raw')
            .select('*')
            .gte('date', julyFourth)
            .order("date_column", desc=False)
            .limit(1)
            .execute())
    past_dif = abs(past-julyFourth)
    fut_dif = abs(future-julyFourth)
    if past_dif > fut_dif:
        print(f"Nearest Date: {past}")
    else:
        print(f"Nearest Date: {future}")