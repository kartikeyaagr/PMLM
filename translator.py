import re

# Dictionary of triggers and their "Real Meaning"
# You can expand this list infinitely.
SubtextDictionary = {
    r"per my last email": "Can you read? I already answered this.",
    r"regards": "I hate you.",
    r"warm regards": "I hate you, but HR is watching.",
    r"thanks in advance": "You have no choice but to do this.",
    r"correct me if i'm wrong": "I am not wrong. I am daring you to challenge me.",
    r"let's take this offline": "I am about to yell at you in private.",
    r"just following up": "Why are you ignoring me?",
    r"as discussed": "Don't try to weasel out of what you promised.",
    r"hope this helps": "Stop asking me stupid questions.",
    r"great catch": "I am embarrassed you found my mistake.",
    r"cc'ing": "I am telling on you to your boss.",
    r"circle back": "I am procrastinating this decision.",
    r"going forward": "Don't ever do that again.",
}


def translate_passive_aggression(text, label, confidence):
    text_lower = text.lower()

    # 1. Check for specific triggers
    for pattern, translation in SubtextDictionary.items():
        if re.search(pattern, text_lower):
            return f'"{translation}"'

    # 2. If no trigger found, give a generic translation based on class & confidence
    if label == "PASSIVE_AGGRESSIVE":
        if confidence > 0.9:
            return '"I am professionally furious."'
        else:
            return '"I am annoyed, but trying to hide it."'

    elif label == "NEUTRAL":
        return '"I am just doing my job. No subtext."'

    elif label == "POSITIVE":
        if "!" in text:
            return '"I am actually happy! (Or caffeinated)."'
        return '"Good job."'

    return '"Unknown intent."'
