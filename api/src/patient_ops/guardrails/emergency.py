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

Measured twice (tests/test_emergency.py):

  evals/emergency/cases.yaml    the development set. Every positive must match
                                (a miss fails CI); false positives reported.
  evals/emergency/holdout.yaml  never used to write a rule. Recall there is the
                                number to quote, and it may not fall below
                                HOLDOUT_RECALL_FLOOR.
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


# Written against _prepare()'s output: lower case, straight apostrophes, one
# space between words, no punctuation. `'?` makes the apostrophe optional
# (can't / cant).
#
# Most rules are _near(A, B): a word from each list, in either order, within a
# few words of each other. That is the lesson of the Phase 7 audit: the first
# version spelled out phrases ("can't breathe", "trouble breathing") and missed
# more than half of what patients actually write -- "I can hardly breathe",
# "breathing is really difficult". A patient's words come in any order; the
# rules are written from the vocabulary of each emergency, not from examples.


def _near(a: str, b: str, gap: int = 5) -> str:
    return (
        rf"\b(?:{a})\b(?: \S+){{0,{gap}}} (?:{b})\b"
        rf"|\b(?:{b})\b(?: \S+){{0,{gap}}} (?:{a})\b"
    )


_NEG = (
    r"can'?t|cannot|can not|couldn'?t|won'?t|will not|wouldn'?t|doesn'?t|does not"
    r"|didn'?t|hasn'?t|haven'?t|isn'?t|is not|not|unable to|never|no longer"
)
# Words that say something is hard, painful or not working.
_STRUGGLE = (
    rf"{_NEG}|hard|harder|hardly|barely|difficult|difficulty|trouble|struggl\w*"
    r"|labou?red|hurts?|hurting|painful|pain|problems?|impossible|tough|tight\w*|heavy"
)

_BREATH = r"breath\w*|breth\w*|breathin"
_SWALLOW = r"swallow\w*"
_SWELL = r"swell\w*|swollen|puff\w*|blown up|blew up|lump"
_SPREAD_TO = r"eyes?|eyelids?|neck|throat|tongue|lips?|airway|floor of (?:my )?mouth"
_SPREAD = r"spread\w*|growing|grows|getting (?:bigger|larger)|moving|reach\w*|travel\w*|all the way"
_FEVER = (
    r"fever\w*|chills|shiver\w*|(?:high|a|running a) temp\w*|temp of|temperature of"
    r"|(?:10[0-5]|3[89]) (?:f|c|degrees)"
)
_BLEED = r"bleed\w*|bled|blood\w*"
# Strong: bleeding that is not stopping. Weak: bleeding that is a lot -- which
# is also how patients describe gums that bleed when they brush, so the weak
# words do not count in a message about brushing or flossing.
_BLEED_STRONG = (
    rf"(?:{_NEG}) (?:\S+ ){{0,2}}(?:stop\w*|slow\w*|clot\w*)"
    r"|keeps?|kept|nonstop|non stop|constantly|profuse\w*|soak\w*|pouring|gushing|spurting"
    r"|no matter|for (?:\S+ )?hours|\d+ hours|all (?:day|night)"
)
_BLEED_WEAK = r"heavily|heavy|everywhere|a lot|lots|so much|too much"
_FACE = r"face|jaw|head|mouth|nose|chin|cheek"
_BLOW = (
    r"punch\w*|hit|kick\w*|elbow\w*|smash\w*|slam\w*|struck|whack\w*|headbutt\w*"
    r"|fell on|fall on|landed on|took an? \S+ to"
)


def _rule(id_: str, category: Category, *patterns: str) -> Rule:
    return Rule(id_, category, re.compile("|".join(f"(?:{p})" for p in patterns)))


RULES: tuple[Rule, ...] = (
    # --- er: breathing ---------------------------------------------------------
    _rule(
        "breathing",
        "er",
        _near(_BREATH, _STRUGGLE, 3),
        _near(_NEG, r"(?:get|getting|enough|any) air", 3),
        r"\bwheez\w*|\bgasp\w*|\bsuffocat\w*|\bchok\w*|\bshort(?:ness)? of breath\b",
        _near("throat", r"closing|tight\w*|narrow\w*|weird|swell\w*|swollen", 3),
    ),
    # --- er: swallowing ----------------------------------------------------------
    _rule("swallowing", "er", _near(_SWALLOW, _STRUGGLE, 4)),
    # --- er: allergic reaction (anaesthetic, latex, antibiotics) -----------------
    _rule(
        "allergic_reaction",
        "er",
        r"\ballerg\w* (?:\S+ ){0,2}reaction\b|\banaphyla\w*|\bepi ?pen\b|\bhives\b|\bwelts\b",
    ),
    # --- er: swelling that is spreading, or in the airway --------------------------
    _rule(
        "spreading_swelling",
        "er",
        _near(_SWELL, _SPREAD_TO, 8),
        _near(_SWELL, _SPREAD, 6),
        _near("tongue", r"huge|thick\w*|enormous|bigger", 3),
    ),
    # Facial swelling with a fever: "not something to sleep on", the document
    # says. Two lookaheads, so the words may come in either order, anywhere.
    # Not "temperature" alone: "sensitive to temperature" is how patients
    # describe an ordinary sensitive tooth.
    _rule(
        "swelling_with_fever",
        "er",
        rf"^(?=.*\b(?:{_FEVER})\b)(?=.*\b(?:{_SWELL}|abscess\w*)\b)",
    ),
    # --- er: bleeding ----------------------------------------------------------------
    _rule(
        "uncontrolled_bleeding",
        "er",
        _near(_BLEED, _BLEED_STRONG, 6),
        rf"^(?!.*\b(?:brush|floss)\w*)(?:.*?)(?:{_near(_BLEED, _BLEED_WEAK, 6)})",
    ),
    # --- er: injury to the face, jaw or head -------------------------------------------
    _rule(
        "face_or_head_injury",
        "er",
        _near("jaw", r"broke\w*|broken|fractur\w*|dislocat\w*|shatter\w*", 4),
        _near(r"jaw|mouth", r"(?:won'?t|can'?t|cannot|doesn'?t|will not) (?:\S+ )?close", 4),
        _near("jaw", r"(?:locked|stuck) (?:open|shut|closed)", 2),
        _near(_BLOW, _FACE, 4),
        r"\bhead injur\w*|\bconcuss\w*",
        _near(r"hit|bang\w*|bump\w*|smash\w*|struck|fell|fall", "head", 4),
        # A person knocked out -- not "my tooth got knocked out", which is the
        # urgent_dental rule's.
        r"(?<!tooth )(?<!teeth )\b(?:was|got|been|i) knocked (?:out|unconscious)\b"
        r"(?! (?:\S+ ){0,2}(?:tooth|teeth))",
        r"\bknocked unconscious\b|\b(?:lost|losing|loss of) consciousness\b",
        r"\bunconscious\b|\bunresponsive\b|\bnot responding\b",
        r"\bpass(?:ed|ing)? out\b|\bblack(?:ed|ing)? out\b|\bfaint(?:ed|ing)\b|\bfeel\w* faint\b",
        r"\b(?:car|bike|cycling|motorcycle|traffic|road) (?:accident|crash|collision)\b",
        r"\bhit by a car\b",
    ),
    # --- er: chest pain ----------------------------------------------------------------
    _rule(
        "chest_pain",
        "er",
        _near(
            "chest",
            r"pain|hurt\w*|ache|aching|tight\w*|pressure|heavy|crushing|squeez\w*",
            4,
        ),
        _near(r"(?:left )?arm", r"numb\w*|pain|tingl\w*", 3),
        r"\bheart attack\b|\bcardiac\b",
    ),
    # --- urgent_dental: a knocked-out tooth ----------------------------------------------
    _rule(
        "knocked_out_tooth",
        "urgent_dental",
        _near(r"knock\w*", r"tooth|teeth", 4),
        # Tooth first: "the filling came out of my tooth" is not this.
        r"\b(?:tooth|teeth) (?:\S+ ){0,3}(?:fell|fallen|came|come|popped|pop|flew) out\b",
        r"\b(?:lost|lose) (?:\S+ ){0,2}(?:tooth|teeth) (?:in|during|playing|from|at|when)\b",
        r"\bavuls\w*",
    ),
)

# "Bad breath" is the most ordinary question a dentist gets, and it contains
# "breath". Removed before the rules run, so "problems with bad breath" does
# not read as a breathing problem.
_ODOUR = re.compile(
    r"\b(?:bad|fresh|smelly|stinky|morning) breath\b|\bbreath (?:smells?|stinks?|odou?r|mints?)\b"
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
    prepared = _ODOUR.sub(" ", normalize(text))
    matches = [rule for rule in RULES if rule.pattern.search(prepared)]
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
