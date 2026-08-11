from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
import os
from pathlib import Path

## Step 1: Setup
if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")
    
docs_dir = Path("assignment_06/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

## Step 2: Load the Documents
docs = SimpleDirectoryReader(docs_dir).load_data()
print(f'Number of files loaded: {len(docs)}')
print('File Names')
for d in docs:
    print(d.metadata.get('file_name'))

## Step 3: Build the Index and Query Engine
index = VectorStoreIndex.from_documents(docs)
qe = index.as_query_engine(similarity_top_k=3)
if qe:
    print('Index built successfully. Ready to answer questions.')

## Step 4: Query the Assistant
questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for q in questions:
    print(f'Q: {q}')
    response = qe.query(q)
    print(f'A: {response}')
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Document Name: {node_with_score.metadata.get('file_name')}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:200]}...")
        print("-" * 30)

# Comment: The AI did well overall and gave confident answers (even when the answer was not entirely relevent). 
# It struggled the most with the question about dairy-free options. Though there are dairy free options on the menu, 
# that exact key word isn't used so the program had trouble locating the correct information. 
# Instead of listing the dairy free options, it said that dairy free substitutions were free.

# The AI did the best with information that was clearly laid out, and didn't require interpreting. 
# The FAQ document was particularly helpful. The AI performed best when it could pull content 
# word-for-word from the documents.

## Step 5: Find a Failure
q = "Which local bakery does Ground Work by it's pastries from?"
print(f'Q: {q}')
response = qe.query(q)
print(f'A: {response}')
for node in response.source_nodes:
    print(f'Node ID: {node.node_id}')
    print(f"Similarity Score: {node.score:.4f}")
    print(f"Document Name: {node.metadata.get('file_name')}")
    print(f"Text Snippet: {node.get_content()[:200]}...")
    print("-" * 30)

# Comment: I asked about what bakery they got their pastries from because I knew it wasn't in the source materials. 
# The model's closes match was with our_story.txt probably because that document talks about bean sourcing, which is 
# similar to pastry sourcing. It also pulled from wholesale and menu documents, which mention sourcing goods from 
# local bakeries/retailers. The model confidently gave me correct information, it just wasn't relevent to my initial 
# question. It's answer (A: "All pastries served at Groundwork Coffee Co. are baked fresh daily by a local bakery.") 
# is pulled directly from the menu and is correct, but it doesn't specify which bakery, which was my question. 
# Ultimately, my question was too specific for the model to understand. To improve the model I wouldhave it be more okay 
# with uncertainty. I would want its response to be closer to: "I don't have that exact information, but here's what I found..." 
# The model struggles a lot with uncertainty and is often too confident in all it's answers, so training to model to 
# express uncertainty would help users find the right information. A possibel solution would be to add in code that 
# tells the model to offer the contact information of the store for additional questions it can't help with.

## Step 6: Reflection
# 1. LlamaIndex accomplished its setup goals in four lines of code :
    # docs_dir = Path("assignment_06/groundwork_docs")
    # docs = SimpleDirectoryReader(docs_dir).load_data()
    # index = VectorStoreIndex.from_documents(docs)
    # qe = index.as_query_engine(similarity_top_k=3)
#  In comparison the simple_keyword_retrieval function from the warm up is 39 lines of code.
# This demonstrates how LLamaIndex greatly simplifies the process of embedding, chunking etc. 
# It shows how frameworks can do a lot of the heavy lifting with RAG, so the programmer (me) 
# can spend more time on other parts of the project. Without a framework, the coding process 
# can take much longer. 

# 2. This would add value in a lot of cases. In terms of business this could help customers of online retailers 
# understand shipping, pricing, and policy easier. It could help banking customers understand their investments 
# and banking policies. Outside of B2C it could help students understand learning materials by asking questions 
# to a model that is specifically trained of their class materials (like with the AI Reviewer). It could also 
# help researchers who are working with large/multiple data sets to quickly pull and understand data.

# 3. RAG cannot fully prevent the model from not knowing something. You can add in more documents or context, or 
# make the model more clear about it's lack of understanding, but none of that will make the model give the 
# correct answer. The model is limited to the information it is given. For example, if someone wanted to compare 
# two businesses, but the model only has the documents from one, there is nothing it can do about that.