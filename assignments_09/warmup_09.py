from datetime import date
from dotenv import load_dotenv
import os
from supabase import create_client

# ----- Supabase Connection ----- #

## Connection Question 1 ##
# Supabase needs a project URL and and API key to connect to my python code. 
# The URL can be found on the dashboard page of the project in Supabase. 
# It's at the top under the name of the project, and there is a dropdown menu to select which piece of
# information user would like to copy. As for API keys, Supabase has two options. One of which is in the 
# same dropdown menu as the URL. That is the public key we are using for this project. Both API keys can 
# be found ender Settings > API Keys. Keeping this information private is how you ensure that only approved 
# people can use it. If this information was public, anyone could log in and alter data.


## Connection Question 2 ##
def get_client():
    load_dotenv()
    if os.getenv("SUPABASE_URL") == None:
        raise ValueError ("Can't locate project URL")
    elif os.getenv("SUPABASE_KEY") == None:
        raise ValueError(("Can't locate project key"))
    else:
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
        response = supabase.table("connection_test").select("*").execute()
        print(response.data)
        return supabase


## Connection Question 3 ##
# For this coure we disabled RLS so that we wouldn't have to deal with access policies or other 
# obstacles to managing data. Mainting RLS is important in real world scenarios containing sensative data. 
# If an online store has all its costumers personal information (address, credit card numbers, phone numbers) 
# maintaining their privacy is a huge concern. Therefore, mainting high level RLS means only a few people can 
# access this information to prevent data leakage. For this project we aren't looking at private data, the chances 
# of hackers trying to steal student homework is pretty low, and access policies become more complex with more data, 
# so it's easiest if our RLS is turned off.

# ----- supabase-py CRUD ----- #

## CRUD Question 1 ##
def insert_test_record(supabase):
    today = str(date.today())
    record = {
        "date": today,
        "temperature_2m_max": 18.9,
        "temperature_2m_min": 14.4,
        "precipitation_sum":  0.0,
        "wind_speed_10m_max": 15,
    }
    response = supabase.table("weather_raw").insert(record).execute()
    print(response.data)

    # Adding this line twice will give an error. Insert raises an error 
    # when the same row is added twice, because their can't be two identical 
    # pieces of information. Unlike with upsert which will update an existing 
    # row. In this instance the existing row won't update, the error will come 
    # from trying to create a new row with the same primary key.

## CRUD Question 2 ##
def get_records_by_date_range(supabase, start, end):
    response = (
        supabase
        .table("weather_raw")
        .select("*")
        .gte("date", start)
        .lte("date", end)
        .execute()
        )
    return response.data

## CRUD Question 3 ##
# Both insert and upsert add data to a table. The issue with insert is that it will raise an error 
# if two rows have the same primary key. Upsert handles this by updating an existing row is the primary 
# key exists, or adding a new row if it doesn't.

# For example, in a college database with student information, things may need to get updated. If a student 
# is signing up for classes for the first time as a freshman a new row will need to be added with a new student 
# id (primary key) and the classes they are set to take. But if a student has already registered for classes 
# they may need to add or drop a course, so their existing data needs to be updated. In this instance 
# upsert is the best solution.

# However, insert may be more useful in instances when you don't want data to change. If a bank uses account 
# numbers as primary keys, but two separate customers have the same account number, this could cause a lot of 
# issues. In this instance the bank would want to ensure the right money is going to the right people. So when 
# adding a new client there should be an error message if the new account number already exists. This would be a 
# good time to use insert.

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict = "date", count="exact")
        .execute()
    )
    print(f"Rows Affected: {response.count}")

# ----- Idempotency ----- #

## Idempotency Question 1 ##
# Indempotency is important because without it you run the risk of adding duplicate data. 
# For our weather project it would be a problem if there are 5 tempuratures for the same date and time. 
# This could crash other parts of code and it wouldn't be an accurate view of the data. 
# It can't be 75 degrees and sunny 5 times over, there can only be one temperature at one place and time.

# For example, if a customer is trying to make a purchase online and they hit the pay button twice because 
# the screen isn't loading fast enough, they shouldn't get charged twice. Indempotency ensures that error 
# doesn't happen.

supabase = get_client()
# insert_test_record(supabase)
response = get_records_by_date_range(supabase, "2026-01-01", "2026-09-02")
print(response)
records = [{
        "date": "2026-08-30",
        "temperature_2m_max": 18,
        "temperature_2m_min": 14,
        "precipitation_sum":  0.0,
        "wind_speed_10m_max": 14,
    },
    {
        "date": "2026-03-30",
        "temperature_2m_max": 10,
        "temperature_2m_min": 11,
        "precipitation_sum":  2.0,
        "wind_speed_10m_max": 11,
    },
    ]
safe_upsert(supabase, records)