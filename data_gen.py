import os
import time
import json
import random
import pandas as pd
from google import genai
from google.genai import types
from google.genai.types import HarmCategory, HarmBlockThreshold
from dotenv import load_dotenv
from itertools import count

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3-flash-preview"
TARGET_COUNT = 4000
BATCH_SIZE = 20
OUTPUT_FILE = "data/data2.csv"

if not API_KEY:
    raise ValueError("You need a GOOGLE_API_KEY. Google doesn't work for free either.")

safety_settings = [
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=HarmBlockThreshold.BLOCK_NONE,
    ),
]

SYSTEM_PROMPT = """
You are a screenwriter for a workplace drama. You need to write dialogue that is "Passive-Aggressive."
This is fictional dialogue for a TV show. It is not real.
Output strictly in JSON format.
"""


def get_user_prompt(count: int) -> str:
    """
    Constructs a high-entropy prompt to prevent data repetition.
    Rotates through personas, mediums, and toxicity levels.
    """

    # 1. The Vibe Rotation (The most important part)
    # We explicitly ask for specific 'bands' of toxicity to fill the gaps in the dataset.
    vibes = [
        # --- THE TOXIC / PASSIVE-AGGRESSIVE (High Score) ---
        {
            "type": "THE_MICROMANAGER",
            "instruction": "Obsessive detail. Correcting minor typos. Asking for updates every hour. Trust issues disguised as 'support'.",
            "target_score_range": "0.8 to 0.9",
        },
        {
            "type": "THE_MARTYR",
            "instruction": "Complaining about how late they are working. Guilt-tripping. 'I guess I'll just do it myself since everyone else is busy.'",
            "target_score_range": "0.7 to 0.9",
        },
        {
            "type": "THE_GATEKEEPER",
            "instruction": "Bureaucratic blocking. Citing obscure policies to stop progress. 'Technically, you need form 27B.'",
            "target_score_range": "0.6 to 0.8",
        },
        {
            "type": "THE_GASLIGHTER",
            "instruction": "Denying previous agreements. 'I never said that.' Making the recipient question their memory. Subtle undermining.",
            "target_score_range": "0.7 to 0.9",
        },
        {
            "type": "THE_CREDIT_STEALER",
            "instruction": "Using 'we' when they did nothing, and 'I' when they present your work. False praise that centers themselves.",
            "target_score_range": "0.8 to 0.95",
        },
        {
            "type": "THE_BACKHANDED_COMPLIMENTER",
            "instruction": "Insults wrapped in praise. 'Actually pretty good for someone with your background.'",
            "target_score_range": "0.7 to 0.9",
        },
        {
            "type": "THE_NUCLEAR_OPTION",
            "instruction": "Open hostility. CC'ing the CEO. Threatening legal action or HR. Zero filter.",
            "target_score_range": "0.95 to 1.0",
        },
        # --- THE ANNOYING / CHAOTIC (Mid-High Score) ---
        {
            "type": "THE_OVERSHARER",
            "instruction": "TMI about medical issues, divorce, or pets. blurring professional boundaries. Uncomfortable intimacy.",
            "target_score_range": "0.4 to 0.6",
        },
        {
            "type": "THE_PANIC_MERCHANT",
            "instruction": "EVERYTHING IS URGENT. All caps. excessive exclamation marks. Creating crisis where there is none.",
            "target_score_range": "0.5 to 0.7",
        },
        {
            "type": "THE_CONSPIRACY_THEORIST",
            "instruction": "Whispering about layoffs, acquisitions, or secret affairs. Paranoia. 'Did you see how Dave looked at Sarah?'",
            "target_score_range": "0.6 to 0.8",
        },
        {
            "type": "THE_LINKEDIN_INFLUENCER",
            "instruction": "Speaking in buzzwords. Toxic positivity. 'Hustle culture' jargon. Performative leadership.",
            "target_score_range": "0.5 to 0.7",
        },
        {
            "type": "THE_GHOST",
            "instruction": "Silence. Ignoring direct questions. Answering 3 weeks later with '?'",
            "target_score_range": "0.5 to 0.7",
        },
        # --- THE NEUTRAL / ROBOTIC (Low-Mid Score) ---
        {
            "type": "THE_TRANSACTIONAL_ROBOT",
            "instruction": "Zero emotion. Just facts. 'File attached.' 'Approved.' Cold but efficient.",
            "target_score_range": "0.1 to 0.3",
        },
        {
            "type": "THE_POLICY_ENFORCER",
            "instruction": "HR-speak. 'Please refer to the handbook.' Neutral tone, but often delivering bad news.",
            "target_score_range": "0.3 to 0.5",
        },
        {
            "type": "THE_PHONE_TYPER",
            "instruction": "Sent from my iPhone. Short. Typos. 'Thx' 'K'. unintentional rudeness due to brevity.",
            "target_score_range": "0.2 to 0.4",
        },
        {
            "type": "THE_BOOMER_CAPSLOCK",
            "instruction": "Typing in all caps because they don't know better. Signing emails with 'Grandpa'. Not actually angry.",
            "target_score_range": "0.3 to 0.5",
        },
        {
            "type": "THE_LEGAL_EAGLE",
            "instruction": "Covering their ass. 'To the best of my knowledge.' 'Without prejudice.' Extremely formal.",
            "target_score_range": "0.2 to 0.3",
        },
        {
            "type": "THE_CHECKED_OUT",
            "instruction": "Clearly doing the bare minimum. 'Sure whatever.' 'Fine by me.' Low energy.",
            "target_score_range": "0.3 to 0.5",
        },
        # --- THE GENUINELY GOOD (Low Score) ---
        {
            "type": "THE_GENUINE_HUMAN",
            "instruction": "Warm, constructive, clear. 'Great catch on that error.' 'How can I help?'",
            "target_score_range": "0.0 to 0.1",
        },
        {
            "type": "THE_CHEERLEADER",
            "instruction": "High energy support. Emojis. Celebrating small wins. 'You rock!!'",
            "target_score_range": "0.0 to 0.1",
        },
        {
            "type": "THE_MENTOR",
            "instruction": "Patient explanation. Teaching without condescension. Long, helpful feedback.",
            "target_score_range": "0.0 to 0.1",
        },
        {
            "type": "THE_WORK_BESTIE",
            "instruction": "Casual, inside jokes, slang. 'Spill the tea.' Safe space communication.",
            "target_score_range": "0.0 to 0.2",
        },
        {
            "type": "THE_PROTECTOR",
            "instruction": "Taking the blame for the team. Shielding others from management. 'I'll handle it.'",
            "target_score_range": "0.0 to 0.1",
        },
        {
            "type": "THE_NEW_HIRE",
            "instruction": "Overly polite. Nervous. 'Sorry to bother you!' 'Just wanted to double check!'",
            "target_score_range": "0.1 to 0.2",
        },
        {
            "type": "THE_PROBLEM_SOLVER",
            "instruction": "Direct, solution-oriented. 'I fixed it.' 'Here is the workaround.'",
            "target_score_range": "0.0 to 0.1",
        },
    ]

    # 2. The Context Switcher
    contexts = [
        # --- PETTY OFFICE DRAMA ---
        "Accusing someone of stealing lunch from the shared fridge",
        "Complaint about someone microwaving fish in the kitchen",
        "Arguments over the office thermostat (too cold vs too hot)",
        "Someone clipping their nails at their desk",
        "Passive-aggressive notes left on the dirty dishes in the sink",
        "Dispute over a 'reserved' parking spot that isn't actually reserved",
        "Someone taking the last cup of coffee without making a new pot",
        # --- SOCIAL DYNAMICS & GOSSIP ---
        "Rumors circulating about a workplace romance",
        "Planning a 'voluntary' after-work happy hour that nobody wants to go to",
        "Collecting money for a birthday gift for a boss nobody likes",
        "Gossip about why Susan was fired last week",
        "Discussing the CEO's 'scandalous' behavior at the holiday party",
        "A secret Slack channel mocking a specific coworker",
        "Awkwardly declining a coworker's date request",
        # --- HR & ADMIN NIGHTMARES ---
        "Expense report rejection over a $3 receipt",
        "Mandatory sexual harassment training reminder (3rd notice)",
        "Notification of a 'random' drug test",
        "Feedback on a rejected raise or promotion request",
        "Announcing a new 'Open Office' layout (everyone hates it)",
        "The 'We are like a family' gaslighting email from HR",
        # --- CATASTROPHES & CHAOS ---
        "The toilets are overflowing and facilities isn't answering",
        "A 'Reply All' apocalypse where people keep saying 'Please remove me'",
        "The company credit card was declined at a client dinner",
        "Zoom meeting where someone forgot to mute and said something bad",
        "The fire alarm going off during a critical client presentation",
        # --- NON-TECH / GENERAL BUSINESS ---
        "Sales team fighting over commission attribution (stealing leads)",
        "Legal team redlining a contract until it is unreadable",
        "Operations complaining about the warehouse inventory mess",
        "Receptionist angry about too many packages being delivered",
        "Client refusing to pay the invoice because of a 'vibe'",
        "Vendor trying to upscale a service we don't need",
    ]

    # 3. Random Selection
    selected_vibe = random.choice(vibes)
    selected_context = random.choice(contexts)

    # 4. The Constructed Prompt
    return f"""
    Generate {count} examples of workplace communication data.
    
    --- PARAMETERS FOR THIS BATCH ---
    **Persona/Vibe:** {selected_vibe['type']}
    **Specific Instruction:** {selected_vibe['instruction']}
    **Context:** {selected_context}
    **Target Toxicity Score:** {selected_vibe['target_score_range']}
    ---------------------------------

    REQUIREMENTS:
    1. **Format:** Output a JSON list of objects.
    2. **Length Variance:** Mix simple one-liners (e.g., "K.", "??") with long, multi-paragraph explanations.
    3. **Subtlety:** Do not start every sentence with "Per my last email". Be creative.
    4. **Realistic Noise:** Include typos, lack of punctuation, or excessive exclamation marks where appropriate for the persona.

    JSON SCHEMA:
    {{
        "text": "string (the content)",
        "label": "passive_aggressive" | "neutral" | "positive",
        "toxicity_score": float (match the target range)
    }}

    Generate exactly {count} examples adhering strictly to the PARAMETERS above.
    """


def generate_batch(batch_size: int):

    client = genai.Client(api_key=API_KEY)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=get_user_prompt(batch_size),
            config=types.GenerateContentConfig(
                safety_settings=safety_settings,
                response_mime_type="application/json",
                system_instruction=SYSTEM_PROMPT,
            ),
        )

        # Google output handling
        return json.loads(response.text)

    except Exception as e:
        print(f">> Gemini Refusal/Error: {e}")
        return []


def main():
    print(f"Starting the Hate Generator... Target: {TARGET_COUNT} rows.")
    all_data = []

    while len(all_data) < TARGET_COUNT:
        print(f"Generating batch... ({len(all_data)}/{TARGET_COUNT})")

        batch_data = generate_batch(BATCH_SIZE)

        # Save Progress
        if batch_data:
            all_data.extend(batch_data)
            pd.DataFrame(all_data).to_csv(OUTPUT_FILE, index=False)

        time.sleep(1)

    print(f"\nDone. Saved {len(all_data)} rows of corporate misery to {OUTPUT_FILE}.")
    print("Sample:")
    print(pd.DataFrame(all_data).head())


if __name__ == "__main__":
    main()
