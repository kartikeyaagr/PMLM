import os
import time
import json
import random
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()

# --- CONFIGURATION ---
API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-sonnet-4-5"
TARGET_COUNT = 4600
BATCH_SIZE = 10
OUTPUT_FILE = "data/data.csv"

if not API_KEY:
    raise ValueError("You need an ANTHROPIC_API_KEY.")

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
        {
            "type": "THE_GASLIGHTER",
            "instruction": "Focus on 'Subtle Shade'. Plausible deniability. Sentences that sound polite but imply the recipient is incompetent.",
            "target_score_range": "0.6 to 0.8",
        },
        {
            "type": "THE_NUCLEAR_OPTION",
            "instruction": "High toxicity. CC'ing the boss, citing policy, demanding immediate explanations. Open hostility disguised as process.",
            "target_score_range": "0.9 to 1.0",
        },
        {
            "type": "THE_CONFUSED_BOOMER",
            "instruction": "Neutral/Ambiguous. Brief, poorly formatted, maybe all caps. hard to tell if angry or just old.",
            "target_score_range": "0.3 to 0.5",
        },
        {
            "type": "THE_GENUINE_HUMAN",
            "instruction": "Actually Positive. Constructive feedback, genuine thanks, clear unburdened communication.",
            "target_score_range": "0.0 to 0.1",
        },
        {
            "type": "THE_TRANSACTIONAL_ROBOT",
            "instruction": "Neutral. Zero emotion. Just file transfers, meeting links, and status updates.",
            "target_score_range": "0.2",
        },
    ]

    # 2. The Context Switcher
    contexts = [
        "Code Review (GitHub comments)",
        "Jira Ticket comments (fighting over scope)",
        "Slack DM (urgent late night requests)",
        "All-hands meeting Q&A submission",
        "HR complaint follow-up",
        "Salary negotiation refusal",
        "Vendor contract termination",
        "Design feedback (making the logo bigger)",
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
        "toxicity_score": float (match the target range),
        "source": "email" | "slack" | "jira"
    }}

    Generate exactly {count} examples adhering strictly to the PARAMETERS above.
    """


def generate_batch(batch_size: int):

    client = anthropic.Anthropic(api_key=API_KEY)

    try:
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": get_user_prompt(batch_size)}],
        )

        # Anthropic output handling
        raw_content = response.content[0].text.strip()

        # Clean up markdown code blocks
        if raw_content.startswith("```json"):
            raw_content = raw_content.replace("```json", "", 1)
        elif raw_content.startswith("```"):
            raw_content = raw_content.replace("```", "", 1)
        if raw_content.endswith("```"):
            raw_content = raw_content.replace("```", "", 1)

        return json.loads(raw_content.strip())

    except Exception as e:
        print(f">> Claude Refusal/Error: {e}")
        return []


def main():
    print(
        f"Starting the Hate Generator (Claude Edition)... Target: {TARGET_COUNT} rows."
    )
    all_data = []

    while len(all_data) < TARGET_COUNT:
        print(f"Generating batch... ({len(all_data)}/{TARGET_COUNT})")

        batch_data = generate_batch(BATCH_SIZE)

        # Save Progress
        if batch_data:
            all_data.extend(batch_data)
            # Create data directory if it doesn't exist to avoid errors
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            pd.DataFrame(all_data).to_csv(OUTPUT_FILE, index=False)

        time.sleep(1)

    print(f"\nDone. Saved {len(all_data)} rows of corporate misery to {OUTPUT_FILE}.")
    print("Sample:")
    print(pd.DataFrame(all_data).head())


if __name__ == "__main__":
    main()
