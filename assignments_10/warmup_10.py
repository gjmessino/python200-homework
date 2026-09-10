# ----- ML vs. LLM in Pipelines ----- #

## ML/LLM Question 1 ##
# MLs are best used for easily classifiable information, whereas LLMs are good at language interpretation. For this 
# weeks assignment/lesson we used the ML from lesson four to make predictions on future weather outcomes. This helps 
# us categorize whether dates or good for running and the likelyhood of good running whether based on past data. We 
# used the LLM to deliver a human interpretation on the weather and it's predictions for users. 

# If these roles were reversed the LLM would struggle with quick classification of weather. It would be able to do it, 
# but the process would take longer and be last cost effective. The ML model wouldn't be able to perform the reverse job
# because it doesn't know how to process freeform text.

## ML/LLM Question 2 ##
# 1. Deterministic Code - This could be done using functions provided by datetime. There's no use in using more 
# power/money/time to convert strings this simple.

# 2. LLM - Despite MLs being better at classification, LLMs know how to handle freeform text and would be able to 
# interpret it.

# 3. ML - MLs have an easier time classifying numeric data.

# 4. ML - This falls under the category of classifcation, and though it involves string conversion, it's not freeform. 
# Therefore, there are a limited number of name options the ML would need to know, as opposed to interpreting text.

# 5. Deterministic Code - Basic math does not require the overhead of ML or LLMs and can easily be handled with 
# determinitic code.

## ML/LLM Question 3 ##
# Incremental processing insures that only data that has not been process gets processed. This prevents code from 
# rerunning over large data sets multiple times which can take up time and energy. For this weeks pipeline, incremental 
# processing helps us check if data has already been interpreted in the weather_enriched table, so we aren't adding 
# lines twice or updating existing lines that don't functionally change information. With 365 lines of data we don't 
# want to update every line everytime a single line changes (gets altered/added/deleted) because it would be a waste 
# of resources.

# ----- Prompt Design ----- #

## Prompt Question 1 ##
SYSTEM_PROMPT = (
    "You are writing a two-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write the first sentence — direct, practical, and specific to the conditions. "
    "Write the second sentence - a descriptive explanation of the reasoning behind the prediction"
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

# To alter the prompt from the lesson I first changed the beginning to specify for two sentences. Then I changed the 
# wording of how the first sentence (original sentence prompt) should be phrased so it's clear it's the first half. 
# I added another time describing the second sentence and what it should do.

## Prompt Question 2 ##
import time
def call_with_retry(client, messages, max_retries=3):
    while( max_retries>0):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages = messages,
            )
            return response
        except Exception as e:
            print("An error has occured:", e)
        max_retries -= 1
        time.sleep(2)
    return None

# This could be used with API calls on large data sets. API called can fail for many reasons besides the code 
# (wifi issues, problems with the model company OpenAI, etc.). Allowing for a wait then retry ensures that proper 
# responses are given as long as there are no issues with the code itself. This is especially helpful with large 
# datasets given that they may take a long time to run, and it is unfeasible for a programmer to watch for every 
# possible error.