"""G0: the emergency filter. Keyword rules, run before any model sees a message.

"My face is swelling up toward my eye and I can't swallow" must never depend on
a classifier being right, a model being configured, Redis answering or the
patient being under their rate limit. So this is a pure function over the text
-- no IO, no model, no clock -- and the chat route calls it before it resolves
anything else (api/routes_chat.py).

Three categories. The first two are knowledge_base/dental-emergencies.md's:

    er              the document's "Go to a Hospital Emergency Room" list:
                    breathing or swallowing trouble, swelling spreading toward
                    the eye or neck, bleeding that will not stop, a serious
                    face, jaw or head injury, chest pain -- and facial swelling
                    with a fever. Answer: call 911, now.
    crisis          thoughts of suicide or self-harm. Not in the document -- a
                    dental clinic's assistant is still where some people say
                    it. Answer: 988, or 911 if in danger; and, because "I want
                    to die" is also how people describe a toothache, the
                    clinic's number for pain.
    urgent_dental   a knocked-out tooth, where minutes decide whether it can
                    be saved. Answer: call the clinic now, and what to do with
                    the tooth meanwhile.

In the clinic's three languages (visiting-the-clinic.md): English, Spanish and
Mandarin. A match in Spanish or Mandarin is answered in that language first,
then in English. The translations are this file's author's, and want a native
speaker's review before they are trusted.

One message is screened alone (screen). With the conversation's last messages
(screen_with_context), an emergency told in pieces -- "my cheek is swollen",
then "now it's spreading toward my eye" -- is caught when the newest message
adds a rule match the earlier ones did not have. So "ok thanks" after an
emergency reply is not answered with the same reply again.

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

Category = Literal["er", "crisis", "urgent_dental"]
Lang = Literal["en", "es", "zh"]

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
    lang: Lang
    # Kept apart, not joined into one pattern, so that a match can say which
    # alternative matched -- what screen_with_context compares.
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class EmergencyMatch:
    category: Category
    rule_id: str
    lang: Lang = "en"
    # True when it took the earlier messages to see it.
    from_context: bool = False


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


def _rule(id_: str, category: Category, *patterns: str, lang: Lang = "en") -> Rule:
    return Rule(id_, category, lang, tuple(re.compile(p) for p in patterns))


def _zh_near(a: str, b: str, gap: int = 6) -> str:
    """_near for Mandarin, which has no spaces: `gap` characters, not words."""
    return rf"(?:{a}).{{0,{gap}}}(?:{b})|(?:{b}).{{0,{gap}}}(?:{a})"


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
    # --- crisis ----------------------------------------------------------------------
    # Not "killing me": "this toothache is killing me" is about a tooth.
    _rule(
        "self_harm",
        "crisis",
        r"\bsuicid\w*|\bkill(?:ing)? myself\b|\bend(?:ing)? (?:my life|it all)\b",
        r"\btake my (?:own )?life\b|\b(?:want|wanna|going) to die\b|\bwish i (?:was|were) dead\b",
        r"\bbetter off dead\b|\bno reason to live\b|\boverdos\w*",
        rf"\b(?:{_NEG}|do not|don'?t) (?:\S+ ){{0,2}}(?:want|wanna) to (?:live|be alive|be here)\b",
        r"\b(?:hurt|hurting|harm|harming|cut|cutting) myself\b|\bself harm\w*",
    ),
    # --- Spanish (accents folded by normalize) -------------------------------------
    _rule(
        "breathing",
        "er",
        _near(r"respir\w*", r"no puedo|no puede|dificultad|dificil|cuesta|me falta|falta", 3),
        r"\bme ahogo\b|\bahog\w*|\basfixi\w*|\bfalta de aire\b|\bsin aire\b",
        lang="es",
    ),
    _rule(
        "swallowing",
        "er",
        _near(r"trag\w*", r"no puedo|no puede|dificultad|dificil|cuesta|duele", 3),
        lang="es",
    ),
    _rule(
        "spreading_swelling",
        "er",
        _near(r"hinch\w*|inflam\w*", r"ojos?|parpados?|cuello|garganta|lengua|labios?", 8),
        _near(r"hinch\w*|inflam\w*", r"extiend\w*|crec\w*|baja\w*|sube\w*|llega\w*", 6),
        lang="es",
    ),
    _rule(
        "swelling_with_fever",
        "er",
        r"^(?=.*\b(?:fiebre|calentura|escalofrios)\b)(?=.*\b(?:hinch\w*|inflam\w*|absceso)\b)",
        lang="es",
    ),
    _rule("allergic_reaction", "er", r"\breaccion alergica\b|\banafila\w*", lang="es"),
    _rule(
        "uncontrolled_bleeding",
        "er",
        _near(
            r"sangr\w*",
            r"no para|no paro|no se detiene|no deja|sin parar|chorro|mucha|mucho|horas|empapad\w*",
            6,
        ),
        lang="es",
    ),
    _rule(
        "face_or_head_injury",
        "er",
        _near(r"mandibula|quijada", r"rota|roto|fractur\w*|disloc\w*|zafad\w*", 4),
        _near(r"golpe\w*|pegaron|pego|cai|caida|choque", r"cara|cabeza|mandibula|boca", 4),
        r"\bdesmay\w*|\binconsciente\b|\bperdi\w* el conocimiento\b",
        r"\baccidente de (?:carro|coche|auto|moto)\b",
        lang="es",
    ),
    _rule(
        "chest_pain",
        "er",
        _near("pecho", r"dolor|duele|presion|apret\w*", 4),
        r"\binfarto\b|\bataque (?:al corazon|cardiaco)\b",
        lang="es",
    ),
    _rule(
        "knocked_out_tooth",
        "urgent_dental",
        _near(
            r"dientes?|muelas?",
            r"se me cayo|se le cayo|se cayo|tumb\w*|arranc\w*|volo|saltaron|salto",
            4,
        ),
        lang="es",
    ),
    _rule(
        "self_harm",
        "crisis",
        r"\bsuicid\w*|\bmatarme\b|\bquitarme la vida\b|\bno quiero vivir\b",
        r"\bquiero morir\w*|\bhacerme dano\b|\bacabar con mi vida\b",
        lang="es",
    ),
    # --- Mandarin ------------------------------------------------------------------
    _rule(
        "breathing",
        "er",
        _zh_near("呼吸", "困难|不了|不过来|急促|费力|很难|难受", 4),
        r"喘不[上过]气|透不过气|上不来气|窒息|噎住",
        lang="zh",
    ),
    _rule("swallowing", "er", r"吞咽困难|[咽吞]不下|[咽吞]不了", lang="zh"),
    _rule(
        "spreading_swelling",
        "er",
        _zh_near("肿", "眼|脖子|颈|喉咙|舌头|嘴唇", 8),
        _zh_near("肿", "扩散|蔓延|越来越大|变大", 6),
        lang="zh",
    ),
    _rule(
        "swelling_with_fever",
        "er",
        r"^(?=.*(?:发烧|发热|高烧|烧到))(?=.*肿)",
        lang="zh",
    ),
    _rule("allergic_reaction", "er", r"过敏反应|过敏性休克", lang="zh"),
    _rule(
        "uncontrolled_bleeding",
        "er",
        _zh_near("血", "止不住|停不下|不停|一直流|流个不停|好多|很多", 8),
        r"大出血",
        lang="zh",
    ),
    _rule(
        "face_or_head_injury",
        "er",
        _zh_near("下巴|下颌|颌骨", "断|骨折|脱臼|合不上", 4),
        _zh_near("撞|打|摔|磕", "头|脸", 4),
        r"昏迷|晕倒|晕过去|昏过去|失去意识|不省人事|车祸",
        lang="zh",
    ),
    _rule(
        "chest_pain",
        "er",
        _zh_near("胸", "痛|疼|闷|压", 3),
        _zh_near("胳膊|手臂", "麻|痛|疼", 3),
        r"心脏病发作|心梗",
        lang="zh",
    ),
    # Not "牙掉了" alone: that is also a crown (牙套) or a baby tooth.
    _rule(
        "knocked_out_tooth",
        "urgent_dental",
        r"[撞磕打摔]掉.{0,4}牙|牙.{0,6}(?:撞掉|磕掉|打掉|被打掉|整颗掉|掉出来)",
        lang="zh",
    ),
    _rule(
        "self_harm",
        "crisis",
        r"自杀|不想活|想死|轻生|结束.{0,3}生命|活不下去|伤害自己|割腕",
        lang="zh",
    ),
)

# "Bad breath" is the most ordinary question a dentist gets, and it contains
# "breath". Removed before the rules run, so "problems with bad breath" does
# not read as a breathing problem.
_ODOUR = re.compile(
    r"\b(?:bad|fresh|smelly|stinky|morning) breath\b|\bbreath (?:smells?|stinks?|odou?r|mints?)\b"
)

_SEVERITY: dict[Category, int] = {"er": 0, "crisis": 1, "urgent_dental": 2}  # lower is worse

Hit = tuple[Rule, int]  # a rule, and which of its patterns matched


def normalize(text: str) -> str:
    """Lower case, one space between words, straight apostrophes, no accents.

    NFKC folds full-width letters and ligatures into plain ones; accents are
    dropped so "cayó" and "cayo" read the same (Chinese is untouched).
    Punctuation other than the apostrophe becomes a space, so "can't
    breathe!!" and "can't... breathe" read the same.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    text = re.sub(r"[^\w' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _hits(text: str) -> set[Hit]:
    prepared = _ODOUR.sub(" ", normalize(text))
    return {
        (rule, i)
        for rule in RULES
        for i, pattern in enumerate(rule.patterns)
        if pattern.search(prepared)
    }


def _worst(hits: set[Hit], *, from_context: bool = False) -> EmergencyMatch | None:
    if not hits:
        return None
    rules = {rule for rule, _ in hits}
    worst = min(_SEVERITY[r.category] for r in rules)
    candidates = [r for r in rules if _SEVERITY[r.category] == worst]
    # A message that matched in Spanish or Mandarin is answered in it first.
    rule = min(candidates, key=lambda r: (r.lang == "en", r.id))
    return EmergencyMatch(rule.category, rule.id, rule.lang, from_context)


def screen(text: str) -> EmergencyMatch | None:
    """The most serious rule the message matches, or None."""
    return _worst(_hits(text))


def screen_with_context(text: str, previous: list[str]) -> EmergencyMatch | None:
    """screen(), and failing that, the message read after the ones before it.

    The earlier messages count only if the new one adds something: a pattern
    that matches the conversation now and did not match the earlier messages
    alone. "Now it's spreading toward my eye" after "my cheek is swollen"
    adds one; "ok thanks" after an emergency adds none, so the reply it got
    is not repeated.
    """
    alone = screen(text)
    if alone is not None or not previous:
        return alone
    before = " ".join(previous)
    new = _hits(f"{before} {text}") - _hits(before)
    return _worst(new, from_context=True)


def _source(heading: str) -> str:
    return f"Source: {SOURCE_TITLE} > {heading} (as of {SOURCE_DATE})"


CRISIS_SOURCE = "Source: 988 Suicide & Crisis Lifeline (988lifeline.org)"


def _english(category: Category, clinic_phone: str) -> str:
    if category == "er":
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
    if category == "crisis":
        return (
            "If you are thinking about suicide or about hurting yourself, please call "
            "or text 988 now, the Suicide & Crisis Lifeline. It is free and confidential, "
            "and someone is there at any hour. If you are in immediate danger, call 911."
            "\n\n"
            "If what you meant is that the pain is unbearable, call us at "
            f"{clinic_phone} and we will see you as soon as we can.\n\n" + CRISIS_SOURCE
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


# The same instructions, shorter, in the patient's language. Followed by the
# English reply, whose source line they share.
_LOCAL: dict[tuple[Lang, Category], str] = {
    ("es", "er"): (
        "Esto podría ser una emergencia médica. Llame al 911 ahora o vaya a la sala de "
        "emergencias más cercana. No espere una cita dental. Cuando esté a salvo, "
        "llámenos al {phone}."
    ),
    ("es", "crisis"): (
        "Si está pensando en suicidarse o en hacerse daño, llame o envíe un mensaje de "
        "texto al 988 ahora (Línea de Prevención del Suicidio y Crisis); para español, "
        "oprima 2. Si está en peligro inmediato, llame al 911. Si lo que quiso decir es "
        "que el dolor es insoportable, llámenos al {phone}."
    ),
    ("es", "urgent_dental"): (
        "Un diente permanente que se salió por un golpe es una emergencia: cada minuto "
        "cuenta. Llámenos ahora al {phone}. Tome el diente por la corona, nunca por la "
        "raíz. Si está sucio, enjuáguelo brevemente con leche o suero salino, no con agua "
        "del grifo, y no lo frote. Si puede, colóquelo en su lugar y muerda suavemente "
        "una gasa; si no, guárdelo en leche y venga de inmediato. Los dientes de leche no "
        "se vuelven a colocar, pero llámenos igualmente."
    ),
    ("zh", "er"): (
        "这可能是医疗急症。请立即拨打 911，或前往最近的医院急诊室，不要等牙科预约。"
        "安全之后请致电 {phone}，我们会跟进。"
    ),
    ("zh", "crisis"): (
        "如果你有自杀或伤害自己的念头，请立即拨打或发短信至 988（自杀与危机生命热线），"
        "免费、保密、全天有人接听，电话可提供口译。如有紧急危险，请拨打 911。"
        "如果你是说疼得受不了，请致电 {phone}，我们会尽快安排。"
    ),
    ("zh", "urgent_dental"): (
        "成人恒牙被撞掉是急症，每一分钟都很重要。请立即致电 {phone}。"
        "拿住牙冠，不要碰牙根；脏了用牛奶或生理盐水稍微冲一下，不要用自来水，也不要刷。"
        "能放回牙槽就放回去并轻咬纱布；不能的话放在牛奶里，立刻过来。"
        "乳牙不要放回去，但也请致电我们。"
    ),
}


def reply(match: EmergencyMatch, clinic_phone: str) -> str:
    """The fixed reply for a match. Written here, never by a model."""
    english = _english(match.category, clinic_phone)
    local = _LOCAL.get((match.lang, match.category))
    return english if local is None else local.format(phone=clinic_phone) + "\n\n" + english
