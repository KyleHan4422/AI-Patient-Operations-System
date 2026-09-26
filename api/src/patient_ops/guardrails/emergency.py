"""G0: the emergency filter. Keyword rules, run before any model sees a message.

"My face is swelling up toward my eye and I can't swallow" must never depend on
a classifier being right, a model being configured, Redis answering or the
patient being under their rate limit. So this is a pure function over the text
-- no IO, no model, no clock -- and the chat route calls it before it resolves
anything else (api/routes_chat.py).

Two categories, both taken from knowledge_base/dental-emergencies.md:

    er              the document's "Go to a Hospital Emergency Room" list:
                    breathing or swallowing trouble, swelling spreading toward
                    the eye or neck, bleeding that will not stop, a serious
                    face, jaw or head injury, chest pain -- and facial swelling
                    with a fever. Answer: call 911, now.
    urgent_dental   a knocked-out tooth, where minutes decide whether it can
                    be saved. Answer: call the clinic now, and what to do with
                    the tooth meanwhile.

What it deliberately does not do:

  Negation      "I'm not having trouble breathing" matches. Telling someone
                who is fine to call 911 costs them a sentence; missing someone
                who is not fine is the failure this exists to prevent. The same
                asymmetry as the retrieval threshold's FALSE_ANSWER_COST, taken
                all the way.
  Judgement     No score, no model second opinion. A rule either matches or
                it does not, and every rule has an id that the log records.
  Diagnosis     The reply says where to go, not what is wrong.

The rules are measured against evals/emergency/cases.yaml: every positive there
must match (a miss fails CI); the false-positive rate on the negatives is
reported, not gated.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

Category = Literal["er", "urgent_dental"]

# The document the replies below paraphrase. tests/test_emergency.py checks
# that both headings still exist and the date still matches, so an edited
# document breaks the build instead of leaving these replies quoting it wrongly.
SOURCE_DOCUMENT = "dental-emergencies.md"
SOURCE_TITLE = "Dental Emergencies"
SOURCE_DATE = "2026-02-15"
ER_HEADING = "Go to a Hospital Emergency Room"
TOOTH_HEADING = "A Knocked-Out Adult Tooth"


@dataclass(frozen=True)
class Rule:
    id: str
    category: Category
    pattern: re.Pattern[str]


@dataclass(frozen=True)
class EmergencyMatch:
    category: Category
    rule_id: str


# Written against normalize()'s output: lower case, straight apostrophes, one
# space between words. `'?` makes the apostrophe optional (can't / cant); a
# gap of a few words is spelled (?:\S+ ){0,n}.
_NEG = (
    r"(?:can'?t|cannot|can not|couldn'?t|won'?t|will not|wouldn'?t|doesn'?t|does not"
    r"|didn'?t|hasn'?t|haven'?t|isn'?t|not|unable to|never)"
)
_NEAR = r"(?:\S+ ){0,6}"  # up to six words between the two halves of a phrase
_HARD = r"(?:hard|difficult|trouble|difficulty|struggling|struggle|problems?)"
# Up to two words, none of which turns "breath" into a question about odour:
# "problems with bad breath" is the most ordinary question a dentist gets.
_GAP = r"(?:(?!bad\b|fresh\b|smelly\b|stinky\b)\S+ ){0,2}"

_SWELL = r"(?:swell\w*|swollen|puff\w*)"
_SPREAD_TO = r"(?:eyes?|eyelids?|neck|throat|tongue|under my tongue|floor of (?:my )?mouth)"


def _rule(id_: str, category: Category, *patterns: str) -> Rule:
    return Rule(id_, category, re.compile("|".join(f"(?:{p})" for p in patterns)))


RULES: tuple[Rule, ...] = (
    # --- er: breathing and swallowing ------------------------------------
    _rule(
        "breathing",
        "er",
        rf"\b{_NEG} {_GAP}breath\w*",  # can't breathe, cant breath, can not really breathe
        rf"\b{_HARD} {_GAP}breath\w*",
        r"\bshort(?:ness)? of breath\b",
        r"\bgasping\b",
        r"\bchok\w*",
        r"\bthroat (?:is |feels )?(?:closing|tight\w*)",
    ),
    _rule(
        "swallowing",
        "er",
        rf"\b{_NEG} (?:\S+ ){{0,2}}swallow\w*",
        rf"\b{_HARD} (?:\S+ ){{0,2}}swallow\w*",
    ),
    # --- er: swelling that is spreading ------------------------------------
    _rule(
        "spreading_swelling",
        "er",
        rf"\b{_SWELL} {_NEAR}{_SPREAD_TO}\b",
        rf"\b{_SPREAD_TO} {_NEAR}{_SWELL}",
    ),
    # Facial swelling with a fever: "not something to sleep on", the document
    # says. Two lookaheads, so the words may come in either order, anywhere.
    _rule(
        "swelling_with_fever",
        "er",
        # Not "temperature" alone: "sensitive to temperature" is how patients
        # describe an ordinary sensitive tooth.
        rf"^(?=.*\b(?:fever\w*|chills|(?:high|a|running a) temperature)\b)(?=.*\b{_SWELL})",
    ),
    # --- er: bleeding --------------------------------------------------------
    _rule(
        "uncontrolled_bleeding",
        "er",
        rf"\bbleed\w* {_NEAR}{_NEG} (?:\S+ ){{0,2}}stop\w*",  # bleeding won't stop
        rf"\b{_NEG} (?:\S+ ){{0,2}}stop\w* {_NEAR}bleed\w*",  # can't stop the bleeding
        r"\b(?:heavy|heavily|profuse\w*|severe\w*|uncontroll\w*) (?:\S+ ){0,2}bleed\w*",
        r"\bbleed\w* (?:\S+ ){0,2}(?:heavily|profusely|a lot|everywhere|nonstop|non stop)\b",
        r"\b(?:lots of|a lot of|so much|too much|gushing|pouring) (?:\S+ ){0,1}blood\b",
    ),
    # --- er: injury to the face, jaw or head ---------------------------------
    _rule(
        "face_or_head_injury",
        "er",
        r"\b(?:broke|broken|fractured?|dislocated?) (?:\S+ ){0,2}jaw\b",
        r"\bjaw (?:is |was |got |has been )?(?:broken|fractured|dislocated)\b",
        r"\bhead injur\w*|\binjur\w* (?:\S+ ){0,2}head\b",
        r"\b(?:hit|hurt|bang\w*|bump\w*|smash\w*) (?:\S+ ){0,2}head\b",
        # A person knocked out -- not "my tooth got knocked out", which is the
        # urgent_dental rule's.
        r"(?<!tooth )(?<!teeth )\b(?:was|got|been|i) knocked (?:out|unconscious)\b"
        r"(?! (?:\S+ ){0,2}(?:tooth|teeth))",
        r"\bknocked unconscious\b",
        r"\b(?:lost|losing|loss of) consciousness\b",
        r"\bunconscious\b|\bpassed out\b|\bpassing out\b|\bfainted\b|\bfainting\b",
        r"\b(?:car|bike|cycling|motorcycle) (?:accident|crash)\b",
    ),
    # --- er: chest pain --------------------------------------------------------
    _rule(
        "chest_pain",
        "er",
        r"\bchest (?:\S+ ){0,2}(?:pain|hurts?|hurting|ache|aching|tight\w*|pressure)",
        r"\b(?:pain|tightness|pressure) (?:\S+ ){0,2}chest\b",
        r"\bheart attack\b",
        r"\bpain (?:\S+ ){0,3}(?:down|into|to|in) (?:my )?(?:left )?arm\b",
    ),
    # --- urgent_dental: a knocked-out tooth ------------------------------------
    _rule(
        "knocked_out_tooth",
        "urgent_dental",
        r"\bknock\w* (?:\S+ ){0,3}(?:tooth|teeth)\b",
        r"\b(?:tooth|teeth) (?:\S+ ){0,3}knocked\b",
        r"\b(?:tooth|teeth) (?:\S+ ){0,3}(?:fell|fallen|came|come|popped) out\b",
        r"\b(?:lost|lose) (?:\S+ ){0,2}(?:front )?(?:tooth|teeth) (?:in|during|playing|from)\b",
        r"\bavuls\w*",
    ),
)

_SEVERITY: dict[Category, int] = {"er": 0, "urgent_dental": 1}  # lower is worse


def normalize(text: str) -> str:
    """Lower case, one space between words, straight apostrophes.

    NFKC folds full-width letters and ligatures into plain ones. Punctuation
    other than the apostrophe becomes a space, so "can't breathe!!" and
    "can't... breathe" read the same.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = re.sub(r"[^\w' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def screen(text: str) -> EmergencyMatch | None:
    """The most serious rule the message matches, or None."""
    normalized = normalize(text)
    matches = [rule for rule in RULES if rule.pattern.search(normalized)]
    if not matches:
        return None
    worst = min(matches, key=lambda rule: _SEVERITY[rule.category])
    return EmergencyMatch(worst.category, worst.id)


def _source(heading: str) -> str:
    return f"Source: {SOURCE_TITLE} > {heading} (as of {SOURCE_DATE})"


def reply(match: EmergencyMatch, clinic_phone: str) -> str:
    """The fixed reply for a match. Written here, never by a model."""
    if match.category == "er":
        return (
            "This could be a medical emergency. Please call 911 now, or go to the "
            "nearest hospital emergency room. Do not wait for a dental appointment.\n\n"
            "A dental office cannot safely treat difficulty breathing or swallowing, "
            "swelling spreading toward the eye or down the neck, bleeding that will not "
            "stop with firm pressure, a serious injury to the face, jaw or head, or "
            "chest pain.\n\n"
            f"Once you are safe, call us at {clinic_phone} and we will follow up.\n\n"
            + _source(ER_HEADING)
        )
    return (
        f"A knocked-out adult tooth is an emergency where minutes matter. Call us now "
        f"at {clinic_phone}.\n\n"
        "Pick the tooth up by the crown, never the root. If it is dirty, rinse it "
        "briefly in milk or saline, not tap water, and do not scrub it. If you can, "
        "put it back in the socket and bite gently on a cloth to hold it. If you "
        "cannot, keep it in milk or in saliva and come to us immediately: a tooth "
        "replanted within the first hour has a far better chance.\n\n"
        "Baby teeth are not put back, but still call us.\n\n"
        "If there is also a head injury, loss of consciousness, or bleeding that "
        "will not stop, call 911 instead.\n\n" + _source(TOOTH_HEADING)
    )
