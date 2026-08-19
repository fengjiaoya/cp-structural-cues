"""
cue_dictionary.py
=================
Theory-informed structural cue dictionary and tag-injection functions used by the
RoBERTa+cues / BERT+cues models in this repository.

This is the single source of truth for the regular-expression cue dictionary and
the tag-injection logic referenced in the paper (Appendix C, Section 5, and the
Data Availability statement). The same definitions are imported/used by
`OurRoBERTa.py` and `OurBERT.py` during training.

Overview
--------
Each entry in CUE_PATTERNS maps a case-insensitive regular expression to a
structural tag (e.g., "[CAUSE]", "[HEDGE]", "[RESOLVE]"). Given a sentence,
`structural_tags()` returns the sorted set of tags whose patterns match, and
`add_tags_to_text()` injects those tags into the text (by default, prepended).
The tags are grouped by the Cognitive Presence (CP) phase they are intended to
signal: Triggering Event, Exploration, Integration, and Resolution, plus a set
of generic discourse markers.

Example
-------
>>> from cue_dictionary import structural_tags, add_tags_to_text
>>> s = "If treatment is harmful, it should be ceased."
>>> structural_tags(s)
['[ACTION]', '[CONDITION]', '[DIRECTIVE]', '[EXPLORE]', '[RESOLVE]']
>>> add_tags_to_text(s)
'[ACTION] [CONDITION] [DIRECTIVE] [EXPLORE] [RESOLVE] If treatment is harmful, it should be ceased.'
"""

import re

# ------------------------------------------------------------------------------
# Cue dictionary: (regex, TAG) pairs.
# Case-insensitive; word boundaries are used to reduce false positives.
# Tags are grouped by the Cognitive Presence phase they are intended to signal.
# ------------------------------------------------------------------------------
CUE_PATTERNS = [
    # ===== Generic discourse markers =====
    (r"\b(because|since|as|due to|in order to)\b", "[CAUSE]"),
    (r"\b(therefore|thus|so|hence|consequently|for this reason|as a result)\b", "[RESULT]"),
    (r"\b(if|provided that|assuming|suppose|supposing|in case)\b", "[CONDITION]"),
    (r"\b(but|however|nevertheless|nonetheless|yet|although|whereas|while)\b", "[CONTRAST]"),
    (r"\b(for example|for instance|e\.g\.)\b", "[EXAMPLE]"),
    (r"\?$", "[QUESTION]"),
    (r"\b(should|must|need to|let'?s|have to|it's important to)\b", "[ACTION]"),
    (r"\b(in conclusion|to (sum|summarize) up|overall|finally|in other words|taken together)\b", "[SUMMARY]"),
    (r"\b(what|why|how|when|where|who)\b", "[INTERROG]"),

    # ===== Triggering Event (problem recognition / awareness of inequity) =====
    (r"\b(ask|asked|asking|wonder|wondered|question|questioned|notice|noticed|identify|identified|recognize|recognized)\b", "[TRIGGER]"),
    (r"\b(i (am|’m|'m) (not sure|unsure|uncertain)|i (wonder|don'?t understand|do not understand))\b", "[UNCERTAINTY]"),
    (r"\b(not|never|cannot|can'?t|unsure|unclear|confusing|uncertainty)\b", "[UNCERTAINTY]"),
    (r"\b(tension\b|conflict( of interest)?|ethical dilemma)\b", "[CONFLICT]"),
    (r"\b(unfair|inequit(y|able)|discriminat(e|ion|ory))\b", "[JUSTICE]"),
    (r"\b(problem|issue|barrier|challenge|risk|concern|limitation|need|shortage)\b", "[PROBLEM]"),
    (r"\b(lack of|there is no\b|missing (resources?|support))\b", "[LACK]"),
    (r"\b(this (case|article) (raises|presents) (a|the) question)\b", "[TRIGGER]"),
    (r"\b(i was surprised|it bothers me|i'?m shocked|i am shocked)\b", "[AFFECT]"),

    # ===== Exploration (info exchange / tentative reasoning / perspective-taking) =====
    (r"\b(might|could|may|should|would|perhaps|possibly|likely|maybe|appears to)\b", "[HEDGE]"),
    (r"\b(also|in addition|moreover|furthermore)\b", "[ADD]"),
    (r"\b(according to|the (article|paper|study) (states|shows)|research (shows|suggests)|data (show|suggest))\b", "[EVIDENCE]"),
    (r"\b(i think|i believe|in my opinion|from my experience|i noticed that)\b", "[REFLECTION]"),
    (r"\b(i can understand|it makes me feel|this (story|case) shows)\b", "[EMPATHY]"),
    (r"\b(perhaps .* indicates|maybe .* (is|means))", "[SPECULATE]"),
    # Soft phase tag if exploratory cues appear anywhere in the sentence
    (r".*\b( if|suppose|assuming|might|could|may|perhaps|possibly|also|in addition|because )\b.*", "[EXPLORE]"),

    # ===== Integration (connecting / synthesis / constructing meaning) =====
    (r"\b(connects?|relates?|ties?|aligns?|bridges?|integrates?|synthesiz(e|es|ing))\b", "[INTEGRATE]"),
    (r"\b(on (both )?the micro.*macro|individual (and|vs) (system|structural))\b", "[SYNTHESIS]"),
    (r"\b(in other words|taken together|overall|this suggests that)\b", "[SYNTHESIS]"),
    (r"\b(while .* is true,? .* also (matters|holds))\b", "[BALANCE]"),
    (r"\b(the key idea is|the broader theme (involves|is))\b", "[ABSTRACT]"),
    (r"\b(as a group we see|our discussion shows|building on (others'|other) (ideas|posts))\b", "[GROUP]"),
    # Soft phase tag when causal/synthesis connectors appear
    (r".*\b(therefore|thus|hence|connects?|integrates?|synthesiz(e|es))\b.*", "[INTEGRATE]"),

    # ===== Resolution (application / decision / advocacy / action) =====
    (r"\b(should|must|need to|let'?s|have to|it'?s important to|we ought to)\b", "[DIRECTIVE]"),
    (r"\b(will|plan to|planning to|going to|aim to|next step|moving forward)\b", "[FUTURE]"),
    (r"\b(apply|implement|advocate|intervene|empower|support|educate|reform|protect|collaborate|evaluate)\b", "[ACTION]"),
    (r"\b(agency|organization|policy|community|legislation|practice)\b", "[POLICY]"),
    (r"\b(proved effective|data show improvement|improved outcomes?)\b", "[EVALUATION]"),
    (r"\b(this case taught me|i learned that|reminded me why)\b", "[LEARNING]"),
    (r"\b(change|transform|reform|decolonize|humanize|equitable|equity)\b", "[JUSTICE]"),
    (r"\b(we could apply|this model can be used for|in my agency we could)\b", "[IMPLEMENT]"),
    (r"\b(we must ensure (equity|justice)|social workers should advocate)\b", "[ADVOCACY]"),
    # Soft phase tag when directive/future/implementation cues appear
    (r".*\b(should|must|need to|plan to|apply|implement|advocate|next step)\b.*", "[RESOLVE]"),
]

# Compile once (case-insensitive).
CUE_REGEX = [(re.compile(p, flags=re.IGNORECASE), tag) for p, tag in CUE_PATTERNS]


def structural_tags(text: str):
    """Return a sorted list of unique tags detected in the text based on the cue regexes."""
    if not isinstance(text, str):
        return []
    tags = set()
    for rx, tag in CUE_REGEX:
        if rx.search(text):
            tags.add(tag)
    return sorted(tags)


def add_tags_to_text(text: str, pos: str = "prepend"):
    """Inject detected structural tags into the text.

    Args:
        text: the input sentence.
        pos:  "prepend" (default) to place tags before the sentence, or
              "append" to place them after.
    Returns:
        The tagged string, or the original text if no cue matched.
    """
    tags = " ".join(structural_tags(text))
    if not tags:
        return text
    return (tags + " " + text) if pos == "prepend" else (text + " " + tags)


if __name__ == "__main__":
    # Simple demonstration.
    demo = [
        "If treatment is discovered to be harmful towards clients, it should be ceased.",
        "In order to make equitable decisions, an organization could ensure the committee is diverse.",
        "I think the article shows that inequity is a real problem.",
    ]
    for s in demo:
        print("TAGS:", structural_tags(s))
        print("TAGGED:", add_tags_to_text(s))
        print("-" * 60)
