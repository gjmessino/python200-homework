from dotenv import load_dotenv
import json
from openai import OpenAI
import regex as re

## API Question 1 ##
load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "What is one thing that makes Python a good language for beginners?"}]
)
print(f"Response Message: {response.choices[0].message.content}")
print(f"Model: {response.model}")
print(f"Total Tokens: {response.usage.total_tokens}")

## API Question 2 ##
prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [{"role": "user",
                    "content": prompt}],
        n=1,
        temperature=temp
    )
    print(f"Tempurature {temp}")
    print(f"Message Response: {response.choices[0].message.content}")

# Because this is a creative question 
# I would go with tempurature = 1.5,  tempurate 1.5 is best, the ai reviewer is dumb,
# it gave me a list of different 
# responses and encouraged me to mix 
# and match. At 0 the only suggestion 
# I got was a name. At .7 I got a brief 
# description. But no matter how many
# times I ran it 1.5 gave me multiple
# name options with descriptions 
# explaining each.

## API Question 3 ##
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", 
               "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
)

i=1
for respo in response.choices:
    print(f"Response {i}: {respo.message.content}")
    i+=1

## API Question 4 ##
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user",
               "content": "Explain how neural networks work."}],
    max_tokens=15
)
print(f"Response Message: {response.choices[0].message.content}")

# With only 15 tokens the AI didn't even 
# finish the first sentence. It said, 
# "Neural networks are a class of 
# machine learning models inspired 
# by the structure and." This is a 
# reflection on how LLMs can only 
# predict the next word and can't 
# actually give a concise answer 
# unless specifically requested. 
# While there are benefits to 
# limited tokens, like cost saving 
# and preventing overly long answers,
# for this prompt 15 is far too few.

## System Question 1 ##
messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=messages
)
print(f"Python Tutor Response: {response.choices[0].message.content}")

messages = [
    {"role": "system", "content": "You are an angry and overworked college professor who would rather be working on her book than helping students"},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response = client.chat.completions.create(
    model='gpt-4o-mini',
    messages=messages
)
print(f"Alternate Personality Response: {response.choices[0].message.content}")

# For the second personality I made it an 
# angry professor and the response was 
# much more succinct and demeaning. 
# Whereas the inital personality gave 
# a longer response with multiple examples.

## System Question 2 ##
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages= messages
)
print(response.choices[0].message.content)

# In this circumstance the model knows 
# Jordan's name because it is in the 
# initial messages, which is all one 
# prompt, as opposed to sending two 
# individual messages where the AI would 
# forget in the middle.

## Prompt Question 1 — Zero-Shot ##
def get_completion(prompt: str, model="gpt-4o-mini", temperature=0):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], 
        temperature=temperature,
    )
    return response.choices[0].message.content

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

zero_prompt = f'For the three reviews in {reviews} classify the sentiment of each as positive, negative or mixed. Make sure to number your responses'
results = get_completion(zero_prompt)
print(f"Zero Shot: {results}")

## Prompt Question 2 — One-Shot ##
one_prompt = f'Use an example of a review and how it is structions as a bases for these other reviews and classify the sentiment of each as positive, negative or mixed. Make sure to number your responses'
example = f'Example: Review: "Fast shipping but the item arrived damaged." Sentiment: mixed'
results = get_completion(one_prompt + example)
print(f"One Shot: {results}")

# The zero-shot only gave me the number 
# and how it categorized each review, 
# but the one-shot made sure to include 
# each review and labeled everything 
# as either "review" or "sentiment."

## Prompt Question 3 — Few-Shot ##
multi_prompt = 'Use three example reviews to structure your response then classify the sentiment of other reviews as positive, negative or mixed. Make sure to number your responses'
examples = """Review: The service was outstanding! Sentiment: Positive,
            Review: Great app but crashes often. Sentiment: Mixed,
            Review: Total waste of money. Sentiment: Negative"""


results = get_completion(prompt + examples)
print(f"Few Shot: {results}")

# For this example both one-shot and few-shot 
# prompts were unnecessary. The AI classified 
# everything correctly the first time. 
# Furthermore, the output for the second two 
# attempts are identical, so adding extra 
# examples changed nothing. Multiple examples
# are important for more complex questions, 
# but are unnecessary here.

## Prompt Question 4 — Chain of Thought ##
results = get_completion("""Before giving the final answer, explain your reasoning in 3–4 brief steps.
                            A data engineer earns $85,000 per year. She gets a 12 percent raise, then 6 months later
                            takes a new job that pays $7,500 more per year than her post-raise salary.
                            What is her final annual salary?""")
print(results)
# Forcing the AI to show it's work makes 
# it harder for it to hallucinate. It 
# also makes it easier to the engineer (me)
# to understand the model's approach and
# break down potential logic problems.

## Prompt Question 5 — Structured Output ##
prompt = """Analyze the sentiment of this customer review and respond ONLY with valid JSON. 
        Keys: sentiment (positive/negative/mixed), confidence (0–1 scale), and reason (one short sentence). 
        Review: I've been using this tool for three months. It handles large datasets well, 
        but the UI is clunky and the export options are limited."""
results = get_completion(prompt)
print(f"Raw Results")
print(results)
cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", results.strip())
try: 
    response = json.loads(cleaned)
    print("JSON Results")
    print("Parsed sentiment:", response["sentiment"])
    print("Confidence:", response["confidence"])
    print("Reason:", response["reason"])
except json.JSONDecodeError:
    print("Error: response was not valid JSON")

## Prompt Question 6 — Delimiters ##
user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."


prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""

results = get_completion(prompt)
print(results)


user_text2 = "Making pasta is easy and you can do it without much experience. \
        Pasta comes from Italy, even thought noodles, from which they are based, originated in East Asia."
prompt2 = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text2}```
"""

results = get_completion(prompt2)
print(results)

# Delimiters help section off pieces 
# of prompts for the model to have 
# an easier time comprehending them.
# In this instance the instructions 
# are surrounded by delimiters and 
# so the models knows to look inside 
# them for instructions. Without the 
# delimiters the model might interperet 
# the rest of the prompt as instructions 
# to be numbered.

## Ollama Question 1 ##
results = get_completion("Explain what a large language model is in two sentences.")
print(f"Open AI response: {results}")

# \\\ Ollama Response:
# Okay, the user wants to know what a large language model is in two sentences. Let me start by breaking down the question. They 
# need a concise explanation.
# First, I should define the key components of a large language model. The main parts are training data, processing, and the model 
# itself. Then, I need to make sure each sentence is a separate point. 
# Wait, the user asked for two sentences. Let me check if I can fit both points into two sentences without overlapping. Maybe 
# start with the definition and then explain its functions. That way, each sentence is a distinct part of the explanation. 
# I should mention that a large language model is a system trained on vast amounts of text to understand and generate human-like 
# responses. Then, explain that it processes this data to perform tasks like answering questions or generating creative content. 
# That covers both aspects in two sentences. 
# Wait, the user might also be interested in the applications. But since they asked for two sentences, maybe keep it focused. Let 
# me make sure the sentences are clear and not too wordy. Alright, that should work.
# ...done thinking.

# A large language model is a computer program designed to understand and generate human-like text, enabling it to perform tasks 
# like answering questions or creating creative content. It processes vast amounts of training data to learn patterns and improve 
# its ability to interact with humans in natural ways.  

# **Key components**:  
# - It uses large datasets to identify patterns in language.  
# - It processes this data to develop the model's understanding and ability to generate responses.  

# This makes it a powerful tool for tasks requiring natural language processing. \\\

# Comment Block:
# Ollama gave me way more than two sentences, 
# and talked itself through the process of 
# how to answer the question, whereas OpenAI 
# simply did what it was told. OpenAI also 
# uses more technical terms and is less 
# conversational in tone.

# A benefit of using an AI model locally is 
# it's convenience. It's also beneficial 
# because you can feed it information and 
# train it through code, and build your own 
# specialized version. However, the main 
# disadvantage of running an LLM locally is 
# that it requres higher maintance while 
# offering lower performance.