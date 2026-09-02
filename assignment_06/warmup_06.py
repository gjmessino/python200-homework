from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
import os
import string

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# ------- RAG Concepts ------- #
## Concepts Question 1
# Scenario A
# Retrieval Augmented Generation (RAG) would be best to keep up with the changing documents every quarter. 
# Given the AI is regularly being given new documents, the lawyers probably only want it to call from the most 
# relevant (often times the most recent documents). RAG would ensure that the AI has a deep knowledge of the 
# law through API calls, while remaining felxible through it's ability to pull specific documents, as opposed 
# to searching everything.

# Scenario B
# Fine tuning is best because there is a finite number of documents for the AI to learn from. 
# Tuning the AI voice to fit the company voice would ensure consistancy. Once the model is fine 
# tuned and trained on the right documents it should replicate the in-house writers consistently.

# Scenario C
# Prompt Engineering is the best solution given the limited number of unique documnets involved. 
# The analyst can add the document to her prompt and make the AI analyze it for this one scenario. 
# The AI doesn't need to remember the document for further use so there is no need for a more complicated or cost heavy measure.

## Concepts Question 2
# If the AI expresses uncertainty than users know to look up information through old fashioned means, 
# or at least consider the lack of knowledge for further considerations in their project. When an AI 
# is confidently wrong, users may take what it says as fact which can be damaging to projects and businesses. 
# An AI's tone helps indicate the usefulness/relevency of the information being given, and can shape 
# how people interact with it. For example, if a reasearcher is given incorrect information through a 
# hallucination, that could mess up their entire study.

## Concepts Question 3

# Wrong Order
steps = [
    "Generate a response from the LLM",
    "Extract text from source documents",
    "Receive the user's query",
    "Retrieve the most relevant chunks",
    "Convert text chunks into embeddings",
    "Inject retrieved chunks into the prompt",
    "Split text into chunks",
    "Embed the user's query",
]

# Right Order (with comments on the side)
steps = [
    "Extract text from source documents", #The AI looks at relevent documents are turns relevent text into strings.
    "Split text into chunks", #The text is broken into smaller pieces making it easier to sort through
    "Convert text chunks into embeddings", #The AI takes the relevent chucks and breaks them into numerical values
    "Receive the user's query", #The users input is sent to the AI so that the AI can break it down and put together an answer.
    "Embed the user's query", #User's input is converted into numerical values to help sort by relevency
    "Retrieve the most relevant chunks", #The AI selects chunks that are deemed to be the most useful.
    "Inject retrieved chunks into the prompt", #The AI takes relevent pieces of text from the chunks
    "Generate a response from the LLM", #AI forms a response and outputs it to the user
]

# ------- Keyword RAG ------- #
def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "your", "our"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

## Keyword Question 1
query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

response = simple_keyword_retrieval(query, documents, True)
print(response)

# Comment: hours.txt was selected as the closest match. To get this result I did have to fiddle with the 
# stop words because the model kept selected "your" as a token, then picking loyalty.txt, so I had to add 
# "your" to the stop word list. Weekends.txt had the use of the word weekends which is why the model selected it.

## Keyword Question 2
query = "Do you have anything without caffeine?"
response = simple_keyword_retrieval(query, documents, True)
print(response)

# Comment: No document was selected. the query tokens ('anything', 'caffeine', 'do', 'have', 'without') 
# weren't available in any of the documents. While the AI has access to the menu, which lists items, 
# there is no mention of caffiene (the most important part of the query). To improve this the AI needs
# access to more information, particularly about what each item on the menu is and if caffination levels. 
# Fine Tuning would be a better solution because it would better prepare the AI for more specific questions.

## Keyword Question 3
query = "How do I sign up for rewards?"
response = simple_keyword_retrieval(query, documents, True)
print(response)

# Prediction: It won't pick anything. The only word from the query present in the document is "for" which is in 
# loyalty.txt. However, the AI is unlikely to use it as a query token. We want the AI to pick loyalty.txt but 
# that is contigent on whether it rates "loyalty" and "rewards" as similar enough words.

# Actual Reponse: I was completely correct. The AI didn't register "for" as one of it's query tokens and despite 
# verbose = True it didn't find anything sufficiently similar enough. To get the correct answer would require using 
# the word "loralty" as opposed to "rewards."

# ------- Semantic RAG Concepts ------- #
## Semantic Question 1
# 1. Vector embedding is the process of taking data and turning it into numerical values so it's relevency can be calculated. 
# This makes it easier to relate (for example) the closeness of two word meanings, given that computers have limited knowledge
# of words and their meanings, but a vast knowledge of math. AI models place these numerical values near of far from each 
# other based on their meaning.

# 2. Cosine similarity measures the closeness of two chunks. With the two values of .85 and .3, .85 is the 
# most relevent, given that cosine is measured on a scale of -1 to 1 with 1 being the most relevent and -1 
# being the complete opposite. The score of .3 shows little similarity, where as .85 demonstrates meanings 
# that are almost the exact same.

# 3. Finding correct chunks despite exact words not being present is possible because of the cosine similarity. 
# Cosine similarity (as describes above) defines the closeness of meanings. In this way that AI can took at 
# synonyms or words that are associated with one another to determine meaning and find relevent chunks.

## Semantic Question 2
# | Feature                    | Keyword RAG                       | Semantic RAG                |
# |----------------------------|-----------------------------------|-----------------------------|
# | What is compared?          | Exact word overlap                | Words with similar meanings |
# | What is retrieved?         | Full document                     | Document Chunks             |
# | Can it handle synonyms?    | No                                | Yes                         |
# | Storage format             | Plain text dictionary             | Vector Embeddings           |
# | Relevance score            | Number of overlapping keywords    | Cosine Similarity           |

# ------- LlamaIndex ------- #
docs = SimpleDirectoryReader("assignment_06/brightleaf_pdfs").load_data()

## LlamaIndex Question 1
index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]
print("\nSimilarity Top K = 3")
for q in questions:
    print(f'\nQ: {q}')
    response = query_engine.query(q)
    print(f'A: {response}')

    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
        print("-" * 30)

# Comment: For both questions the nodes pulled are relevent, particularly the top node pulls 
# for each which have similarity scores of .91 and .88 respectively. 
# The tone of the model is confident for each response. 
# They don't show any signs of doubt or hesitation. 
# The snippets received all appear to be relevent to the questions.

## LlamaIndex Question 2
print("\nSimilarity Top K = 1")
query_enginek5 = index.as_query_engine(similarity_top_k=1)
for q in questions:
    print(f"\nQ: {q}")
    response = query_enginek5.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)

print("\nSimilarity Top K = 5")
query_enginek1 = index.as_query_engine(similarity_top_k=5)
for q in questions:
    print(f"\nQ: {q}")
    response = query_enginek1.query(q)
    print("A:", response)
    
    for node_with_score in response.source_nodes:
        print(f"Node ID: {node_with_score.node.node_id}")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
        print("-" * 30)

# Comment: Making k larger created shorter answers, where k=1 the responses had the most detail. 
# Giving the AI too many samples made answers more vague because it had more information tha it needed.

## LlamaIndex Question 3
question = "If my solar panels create extra energy can I sell in back to brightleaf?"
print(f"\nQ: {question}")
response = query_enginek5.query(question)
print("A:", response)
    
for node_with_score in response.source_nodes:
    print(f"Node ID: {node_with_score.node.node_id}")
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:100]}...")
    print("-" * 30)

# Prediction: The AI is not going to know how to answer my exact question and is going to give me similar 
# information. It will probably pull from product_specs to give me more info about the product or from partnerships.

# Actual Response: It told me there is no information on my question, as opposed to hallucinating or 
# giving me similar but different information.

# Comment: I'd probably make a simpler query, given I made this intentionally hard. 
# To fix this I'd give the AI access to more resources, such as how other companies 
# handle similar queries.

## LlamaIndex Question 4
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)

q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)
print(f'\nQ: {q}')
print(f'A: {response}')

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

q = "How many solar pannels does brightleaf have?"
response = query_engine.query(q)
print(f'\nQ: {q}')
print(f'A: {response}')

faithfulness_result = faithfulness_evaluator.evaluate_response(query=q, response=response)
print("Faithfulness Evaluation: " + str(faithfulness_result.score))

relevancy_result = relevancy_evaluator.evaluate_response(query=q, response=response)
print("Relevancy Result: " + str(relevancy_result.score))

# Comment: Both faithfulness and relevency have only two responses: yes (1) or no (1). 
# Faithfulness tests whether the response is faithful to the documentation. 
# So a 1 means that it received information from the documents, and 0 means it did 
# not recieve information from the documents, and might additionally be hallucinating. 
# As for relevency, that checks whether the response is related to the original query. 
# So you could have a passing faithfulness score and failing relevency if the reponse 
# is accurate, but irrelevent to the question. Or you could have a passing relevency 
# score because it is related to the question but the answer is entirely hallucinated.

# The initial query recieved 1s for both faithfulness and relevency, becuase the answer 
# was relevent to both the query and the douments. For my second query failthfulness = 1 
# but relevency = 0, because the AI could generate a response but not one that answered 
# the question.

# The "LLM-as-a-judge" approach is the combination of relevency and faithfulness tests. 
# It is performed by an LLM outside of our agent to increase objectivity. LLM judges are 
# faster than humans at evaluating if information is correct or not, and can carry this 
# out within the code itself, like what we did above. It's better than other accuracy 
# metrics because it's looking at relevancy, as opposed to other more easily qualifiable data.