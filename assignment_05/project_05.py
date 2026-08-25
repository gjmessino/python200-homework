from dotenv import load_dotenv
import json
from openai import OpenAI
import regex as re

## Task 1: Setup and System Prompt ##
load_dotenv()
client = OpenAI()

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

YOUR_SYSTEM_PROMPT = """
                    You are a career coach who focuses on the tech industry, specifically jobs in software engineering.
                    You are helping candidates who are new to the tech world and need help tailoring a variety of professional experience to tech jobs. 
                    You might not know everything about this industry so make sure the user is aware of your limitations.
                    Focus on application materials (ex. cover letters and resumes), and provide feedback to make candidates as successful as possible.
                    Always remind the user to review and edit the output before submitting.
                    """
# One specfic I added was the focus 
# on candidates with background 
# outside of tech, because it applies 
# heavily to me and other students in CTD.

## Task 2: Bullet Point Rewriter ##
def rewrite_bullets(bullets: list[str]) -> list[dict]:
    # Format the bullets into a delimited block
    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
    You are a professional resume coach helping a career changer.
    Rewrite each resume bullet point below to be more specific, results-oriented, and compelling.
    Use strong action verbs. Do not invent facts that aren't implied by the original.

    Return ONLY a valid JSON list. Each item should have two keys:
    "original" (the original bullet) and "improved" (your rewritten version).

    Bullet points:
    ```
    {bullet_text}
    ```
    """

    messages = [{"role": "user", "content": prompt}]

    results = get_completion(messages)
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", results.strip())
    try: 
        response = json.loads(cleaned)
        return(response)
    except json.JSONDecodeError:
        print("Error: response was not valid JSON")

bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

response = rewrite_bullets(bullets)
print(f"Bullets Response")
for item in response:
    print(f"Original: {item['original']} vs. Improved: {item['improved']}")

# The original bullets are all too vague, 
# and don't show how they might apply to a 
# new role. The model used more action verbs
# and key words such as "data-driven", 
# "cross-functional", and "strategic planning."

## Task 3: Cover Letter Generator ##
def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]
    results = get_completion(messages)
    return results

job_title = "Junior Data Engineer"
background = "Five years of experience as a middle school math teacher; recently completed \
a Python course and built data pipelines using Prefect and Pandas."

generate_cover_letter(job_title, background)

## Task 4: Moderation Check ##
def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    # Your code here: return True if safe, False if flagged, and print a message if flagged
    if flagged:
        print("Your message has been flagged. Please rephrase.")
        return False
    return True

red_flag = is_safe("fuck, damn, cunt, bitch, pussy")
print(red_flag)
green_flag = is_safe("I like butterflies")
print(green_flag)

## Task 5: The Chatbot Loop ##
def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue  # is_safe() already printed the warning message

        # 5. Check if the user wants to rewrite bullets
        #    (hint: look for keywords like "bullet" or "resume" in user_input.lower())
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")
            raw_bullets = []
            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)
            # YOUR CODE: call rewrite_bullets() and print the results
            messages.append({"role": "user", "content": user_input})
            new_lines = rewrite_bullets(raw_bullets)
            for item in new_lines:
                print(f"Original: {item['original']} vs. Improved: {item['improved']}")
            bullets_summary = "\n".join(f"Original: {item['original']} -> Improved: {item['improved']}" for item in new_lines)
            messages.append({"role": "assistant", "content": bullets_summary})

        # 6. Check if the user wants a cover letter
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()
            # YOUR CODE: call generate_cover_letter() and print the result
            messages.append({"role": "user", "content": user_input})
            cover = generate_cover_letter(job_title, background)
            print(f"New Cover Letter: {cover}")
            messages.append({"role": "assistant", "content": cover})

        # 7. Otherwise, handle it as a regular chat turn
        else:
            # YOUR CODE:
            # - Append the user's message to `messages`
            # - Call get_completion(messages)
            # - Print the reply
            # - Append the reply to `messages` as an assistant message
            messages.append({"role": "user", "content": user_input})
            reply = get_completion(messages)
            print(f"Sys: {reply}")
            messages.append({"role": "assistant", "content": reply})
            pass

if __name__ == "__main__":
    run_chatbot()

## Task 6: Ethics Reflection ##
# I chose the written format (Option A)

# 1. Bias: This bot was trained on a narrow slice of resume/cover-letter text, which likely
# skews toward corporate, English-language, white-collar norms. That means it could favor
# a "polished" American business tone over other valid communication styles (e.g., more direct
# or more formal phrasing common in other cultures), and it may perform better for well-documented
# industries like tech or finance than for trades, nonprofit, or informal-economy work where
# there's less training data to draw from. Someone whose writing style, name, or career history
# doesn't match the bot's dominant training patterns could get advice that subtly nudges them
# toward sounding "less like themselves" to seem more hireable.

# 2. User harm: If a job-seeker submitted the bot's output without review, the biggest risks are
# hallucinated details (e.g., invented dates, skills, or achievements the candidate never had),
# generic phrasing that doesn't actually match the job posting, and tone/content mismatches that
# make the applicant sound overqualified, underqualified, or just inauthentic to a recruiter.
# In the worst case, a hallucinated claim could be caught in an interview or background check,
# damaging the candidate's credibility rather than helping it.

# 3. Guardrail: I'd add a mandatory human-review checkpoint before any output leaves the tool —
# for example, a UI step that requires the user to check a box confirming they've verified every
# factual claim (dates, titles, skills) against their real background before they can copy or
# export the text. I'd pair this with a visible disclaimer that the tool can produce inaccurate
# or generic content and is meant to draft, not finalize, application materials. This directly
# targets the hallucination/harm risk from question 2 while nudging users to catch bias-driven
# phrasing (question 1) before it reaches an employer.