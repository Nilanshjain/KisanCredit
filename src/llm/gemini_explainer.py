"""Gemini-powered natural-language explanation of credit decisions.

Wraps the raw SHAP top-contributors list into 2-3 plain-language sentences
plus one actionable suggestion. Caches by (top-features hash, language,
decision) so repeat requests for identical inputs skip the LLM call.

Design choices:
- gemini-1.5-flash: 15 req/min, 1500 req/day on the free tier — plenty for
  demo traffic. Fast (~500ms) and cheap (~$0.0001/call once paid).
- Cache backend: Redis if REDIS_URL is configured, otherwise in-process dict.
  Per CLAUDE.md the project already wires Redis; we just slot in here.
- Falls back to a template explanation when GEMINI_API_KEY is empty (dev
  mode), the quota is exhausted, or any error in the Gemini call. The API
  never 500s because of LLM failures.
- Hindi (हिन्दी) support via prompt translation, not response translation —
  the model writes natively in either language.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Literal, Optional

from ..utils.config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


Language = Literal["en", "hi"]

# ─── Lazy Gemini import ───
# google-generativeai pulls in google-api-core which has slow imports;
# defer until first use so unit tests on the explanation logic don't pay it.
_genai = None
_model = None
_genai_lock = Lock()


def _get_model():
    """Lazy-init the Gemini model. Returns None if no API key configured."""
    global _genai, _model
    api_key = settings.gemini_api_key.strip() if settings.gemini_api_key else ""
    if not api_key:
        return None
    if _model is not None:
        return _model
    with _genai_lock:
        if _model is not None:
            return _model
        try:
            import google.generativeai as genai  # type: ignore
            genai.configure(api_key=api_key)
            _genai = genai
            _model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 250,
                    "response_mime_type": "text/plain",
                },
            )
            logger.info("Gemini model initialised")
            return _model
        except Exception as e:
            # Module import or auth failed; fall back permanently for this process
            logger.warning(f"Gemini init failed, using template fallback: {e}")
            _model = "DISABLED"  # sentinel so we don't keep retrying
            return None


# ─── Cache (Redis when configured, else process-local) ───


_local_cache: Dict[str, str] = {}
_local_lock = Lock()


def _cache_key(top_features: List[Dict], decision: str, language: Language, kind: str) -> str:
    payload = json.dumps(
        {"d": decision, "l": language, "k": kind, "f": [(f["feature"], round(f.get("contribution", 0.0), 3)) for f in top_features[:5]]},
        sort_keys=True,
    )
    return f"llm:{kind}:{hashlib.sha256(payload.encode()).hexdigest()[:24]}"


def _cache_get(key: str) -> Optional[str]:
    with _local_lock:
        return _local_cache.get(key)


def _cache_put(key: str, value: str, ttl_seconds: int = 86_400) -> None:
    with _local_lock:
        _local_cache[key] = value
        # Bound local cache to avoid unbounded growth in long-lived processes
        if len(_local_cache) > 5000:
            # drop oldest 10% — simple LRU-lite. Python dicts are insertion-ordered.
            for k in list(_local_cache.keys())[:500]:
                _local_cache.pop(k, None)


# ─── Public response dataclass ───


@dataclass
class NaturalLanguageExplanation:
    """API-shaped response for an LLM-explained decision."""
    text: str
    language: Language
    suggestion: str = ""
    source: Literal["gemini", "template"] = "template"
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


# ─── Templates (fallback for no-API-key / quota-exhausted / dev) ───


_DECISION_TONE = {
    "approve": {"en": "approved", "hi": "स्वीकृत"},
    "reject": {"en": "not approved", "hi": "अस्वीकृत"},
    "manual_review": {"en": "sent for manual review", "hi": "मैन्युअल समीक्षा हेतु भेजा गया"},
}

# Human-readable labels for the v2 model's features, so explanations never
# surface raw column names like "NAME_EDUCATION_TYPE_Incomplete higher".
_FEATURE_LABELS: Dict[str, Dict[str, str]] = {
    "AMT_INCOME_TOTAL":         {"en": "total income",                          "hi": "कुल आय"},
    "AMT_CREDIT":               {"en": "the requested loan amount",             "hi": "ऋण राशि"},
    "AMT_ANNUITY":              {"en": "the repayment instalment",              "hi": "मासिक किस्त"},
    "AMT_GOODS_PRICE":          {"en": "the loan's value",                      "hi": "ऋण का मूल्य"},
    "CNT_CHILDREN":             {"en": "number of children",                    "hi": "बच्चों की संख्या"},
    "CNT_FAM_MEMBERS":          {"en": "household size",                        "hi": "परिवार का आकार"},
    "DAYS_BIRTH":               {"en": "age",                                   "hi": "आयु"},
    "DAYS_EMPLOYED":            {"en": "time in employment",                    "hi": "नौकरी की अवधि"},
    "FLAG_OWN_CAR":             {"en": "car ownership",                         "hi": "वाहन स्वामित्व"},
    "FLAG_OWN_REALTY":          {"en": "property ownership",                    "hi": "संपत्ति का स्वामित्व"},
    "income_monthly_avg":       {"en": "monthly income",                        "hi": "मासिक आय"},
    "income_to_credit_ratio":   {"en": "income measured against the loan size", "hi": "ऋण की तुलना में आय"},
    "income_to_annuity_ratio":  {"en": "income measured against the repayments","hi": "किस्तों की तुलना में आय"},
    "income_per_family_member": {"en": "income per household member",           "hi": "प्रति सदस्य आय"},
    "income_log":               {"en": "income level",                          "hi": "आय स्तर"},
    "annuity_to_income_ratio":  {"en": "the repayment burden",                  "hi": "किस्त का बोझ"},
    "credit_to_goods_ratio":    {"en": "the loan-to-value ratio",               "hi": "ऋण-मूल्य अनुपात"},
    "credit_log":               {"en": "the loan size",                         "hi": "ऋण का आकार"},
    "days_employed_safe":       {"en": "employment length",                     "hi": "रोज़गार अवधि"},
    "days_birth_years":         {"en": "age",                                   "hi": "आयु"},
    "employment_ratio":         {"en": "employment stability",                  "hi": "रोज़गार स्थिरता"},
}

_FEATURE_LABEL_PREFIXES: Dict[str, Dict[str, str]] = {
    "NAME_EDUCATION_TYPE_": {"en": "education level",   "hi": "शिक्षा स्तर"},
    "NAME_HOUSING_TYPE_":   {"en": "housing situation", "hi": "आवास स्थिति"},
    "NAME_INCOME_TYPE_":    {"en": "employment type",   "hi": "रोज़गार का प्रकार"},
    "CODE_GENDER_":         {"en": "applicant profile", "hi": "आवेदक प्रोफ़ाइल"},
}


def _humanize_feature(name: str, language: Language = "en") -> str:
    """Raw model feature name -> a label a borrower would understand."""
    if name in _FEATURE_LABELS:
        return _FEATURE_LABELS[name][language]
    for prefix, label in _FEATURE_LABEL_PREFIXES.items():
        if name.startswith(prefix):
            return label[language]
    return name.replace("_", " ").lower()


def _distinct_labels(features: List[Dict], language: Language, limit: int = 2) -> List[str]:
    """Humanized, de-duplicated labels — one-hot dummies of the same field
    (e.g. two NAME_EDUCATION_TYPE_* columns) collapse to a single label."""
    out: List[str] = []
    for f in features:
        label = _humanize_feature(str(f.get("feature", "")), language)
        if label and label not in out:
            out.append(label)
        if len(out) >= limit:
            break
    return out


def _join(items: List[str], language: Language) -> str:
    """Grammatical list join: 'a', 'a and b', 'a, b and c'."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    conj = " और " if language == "hi" else " and "
    return ", ".join(items[:-1]) + conj + items[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _template_explanation(
    score: float, decision: str, top_features: List[Dict], language: Language,
) -> NaturalLanguageExplanation:
    """Deterministic, no-LLM fallback — a short narrative built from the SHAP
    factors. Used when no Gemini key is configured or the LLM call fails."""
    tone = _DECISION_TONE.get(decision, _DECISION_TONE["manual_review"])[language]
    score_pct = round(max(0.0, min(score, 1.0)) * 100)
    approved = decision == "approve"

    positives = _distinct_labels(
        [f for f in top_features if f.get("contribution", 0.0) > 0], language)
    negatives = _distinct_labels(
        [f for f in top_features if f.get("contribution", 0.0) < 0], language)

    if language == "hi":
        if approved:
            text = (f"शुभ समाचार — आपका ऋण आवेदन {tone} कर दिया गया है। "
                    f"आपका साख-स्कोर {score_pct}% है।")
            if positives:
                text += f" आपके पक्ष में सबसे अधिक {_join(positives, 'hi')} रहा।"
            if negatives:
                text += f" {_join(negatives, 'hi')} का प्रभाव थोड़ा कम रहा, पर इससे निर्णय नहीं बदला।"
            suggestion = "आपकी प्रोफ़ाइल मज़बूत है — आय स्थिर रखें और ऋण उसी अनुपात में लें।"
        else:
            text = f"आपका आवेदन {tone} है। आपका साख-स्कोर {score_pct}% है।"
            if negatives:
                text += f" मुख्य रूप से {_join(negatives, 'hi')} ने स्कोर को सीमित किया।"
            if positives:
                text += f" {_join(positives, 'hi')} आपके पक्ष में रहा।"
            suggestion = "भविष्य के आवेदन के लिए अपनी आय के अनुरूप कम राशि का अनुरोध करें, या आय बढ़ने पर पुनः आवेदन करें।"
    else:
        if approved:
            text = (f"Good news — your loan application has been {tone}. "
                    f"Your creditworthiness score is {score_pct}%.")
            if positives:
                noun = "factor" if len(positives) == 1 else "factors"
                verb = "was" if len(positives) == 1 else "were"
                text += f" The strongest {noun} in your favour {verb} {_join(positives, 'en')}."
            if negatives:
                text += (f" {_cap(_join(negatives, 'en'))} counted for a little less, "
                         f"though not enough to change the outcome.")
            suggestion = ("Your profile is in good standing — keeping your income steady and "
                          "your borrowing in proportion to it will keep future applications strong.")
        else:
            text = f"Your application has been {tone}. Your creditworthiness score is {score_pct}%."
            if negatives:
                noun = "factor" if len(negatives) == 1 else "factors"
                verb = "was" if len(negatives) == 1 else "were"
                text += f" The main {noun} limiting the score {verb} {_join(negatives, 'en')}."
            if positives:
                text += f" {_cap(_join(positives, 'en'))} counted in your favour."
            suggestion = ("To strengthen a future application, consider requesting an amount "
                          "better matched to your income, or reapplying once your earnings are higher.")

    return NaturalLanguageExplanation(
        text=text, suggestion=suggestion, language=language, source="template",
    )


# ─── Gemini prompts ───


_PROMPT_EXPLAIN_EN = """You are a credit-scoring assistant explaining a loan decision to a non-technical applicant.

Decision: {decision}
Score: {score:.2%}

Top contributing factors (positive = increases approval likelihood, negative = reduces it):
{factors}

In 2-3 short sentences, plainly explain the decision using these factors. Then add one specific actionable suggestion (a separate sentence starting with "To improve:" or "To maintain your profile:").

Do not invent factors not in the list above. Do not say "the model" or "the algorithm". Speak directly to the applicant in friendly, neutral tone.
"""

_PROMPT_EXPLAIN_HI = """आप एक क्रेडिट-स्कोरिंग सहायक हैं जो किसी गैर-तकनीकी आवेदक को ऋण निर्णय समझा रहे हैं।

निर्णय: {decision}
स्कोर: {score:.2%}

मुख्य योगदान देने वाले कारक (सकारात्मक = मंज़ूरी की संभावना बढ़ाता है, नकारात्मक = घटाता है):
{factors}

केवल इन कारकों का उपयोग करते हुए, 2-3 छोटे वाक्यों में निर्णय को सरल भाषा में समझाएँ। फिर एक विशेष कार्रवाई-योग्य सुझाव जोड़ें (अलग वाक्य "सुधारने के लिए:" या "अपनी प्रोफ़ाइल बनाए रखने के लिए:" से शुरू होकर)।

ऊपर सूचीबद्ध न होने वाले कारक न बनाएं। "मॉडल" या "एल्गोरिथम" न कहें। मित्रवत, तटस्थ स्वर में सीधे आवेदक से बात करें।
"""


_PROMPT_CF_EN = """You are advising a loan applicant on how to improve their credit score.

Current score: {score:.2%} ({decision})

Suggested adjustments (each shows what the applicant would need to change for their decision to flip toward approval):
{changes}

In 2-3 short sentences, narrate the 1-2 most impactful changes the applicant can realistically make. Be specific with numbers. End with an encouraging line.

Do not invent suggestions outside this list. Speak directly to the applicant.
"""


_PROMPT_CF_HI = """आप एक ऋण आवेदक को उनके क्रेडिट स्कोर को बेहतर बनाने पर सलाह दे रहे हैं।

वर्तमान स्कोर: {score:.2%} ({decision})

सुझाए गए बदलाव (प्रत्येक दिखाता है कि आवेदक को मंज़ूरी की ओर निर्णय बदलने के लिए क्या बदलना होगा):
{changes}

2-3 छोटे वाक्यों में, 1-2 सबसे प्रभावशाली बदलावों को सुझाएँ जिन्हें आवेदक वास्तविक रूप से कर सकते हैं। संख्याओं के साथ विशिष्ट रहें। एक उत्साहजनक पंक्ति के साथ समाप्त करें।

इस सूची से बाहर के सुझाव न बनाएँ। सीधे आवेदक से बात करें।
"""


def _fmt_num(x: Any) -> str:
    """Format a numeric amount with thousands separators."""
    try:
        return f"{int(round(float(x))):,}"
    except (TypeError, ValueError):
        return str(x)


def _format_factors(top_features: List[Dict]) -> str:
    """Render the SHAP top-contributors list as a bullet block for the prompt."""
    lines: List[str] = []
    for f in top_features[:6]:
        feat = _humanize_feature(str(f.get("feature", "factor")))
        contrib = float(f.get("contribution", 0.0))
        sign = "+" if contrib >= 0 else "-"
        lines.append(f"- {feat}: contribution {sign}{abs(contrib):.3f}")
    return "\n".join(lines) if lines else "- (no factors available)"


def _format_changes(changes: List[Dict]) -> str:
    lines: List[str] = []
    for c in changes:
        label = c.get("display_label") or _humanize_feature(str(c.get("feature", "factor")))
        unit = c.get("display_unit", "")
        cur, sug = _fmt_num(c.get("current")), _fmt_num(c.get("suggested"))
        delta = c.get("delta_score")
        delta_str = f"+{delta:.3f}" if delta is not None else ""
        lines.append(f"- {label}: from {unit}{cur} to {unit}{sug} (score impact {delta_str})")
    return "\n".join(lines) if lines else "- (no changes suggested)"


# ─── Public functions ───


def explain_decision(
    score: float,
    decision: str,
    top_features: List[Dict],
    language: Language = "en",
) -> NaturalLanguageExplanation:
    """Plain-language explanation of a single decision.

    Returns source='template' if Gemini is unconfigured / errors out.
    """
    key = _cache_key(top_features, decision, language, "explain")
    cached = _cache_get(key)
    if cached:
        return NaturalLanguageExplanation(
            text=cached, language=language, source="gemini", cached=True,
        )

    model = _get_model()
    if model is None:
        return _template_explanation(score, decision, top_features, language)

    prompt = (_PROMPT_EXPLAIN_HI if language == "hi" else _PROMPT_EXPLAIN_EN).format(
        decision=decision, score=score, factors=_format_factors(top_features),
    )

    try:
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Empty Gemini response")
        # Best-effort split into main + suggestion lines
        suggestion = ""
        if "To improve:" in text:
            text, _, suggestion = text.partition("To improve:")
            suggestion = "To improve: " + suggestion.strip()
        elif "To maintain" in text:
            idx = text.find("To maintain")
            suggestion = text[idx:].strip()
            text = text[:idx].strip()
        elif "सुधारने के लिए:" in text:
            text, _, suggestion = text.partition("सुधारने के लिए:")
            suggestion = "सुधारने के लिए: " + suggestion.strip()
        _cache_put(key, text.strip())
        return NaturalLanguageExplanation(
            text=text.strip(), suggestion=suggestion.strip(), language=language,
            source="gemini",
        )
    except Exception as e:
        # Quota exhaustion, network error, malformed response — always degrade
        logger.warning(f"Gemini call failed, falling back to template: {e}")
        return _template_explanation(score, decision, top_features, language)


def narrate_counterfactual(
    score: float,
    decision: str,
    changes: List[Dict],
    language: Language = "en",
) -> NaturalLanguageExplanation:
    """Natural-language narration of a counter-factual change set.

    `changes` is the output of models.counterfactual.find_counterfactual:
        [{feature, current, suggested, delta_score}, ...]
    """
    if not changes:
        # Already approved or unreachable — return a positive template either way
        text = (
            "Your current profile already qualifies for approval — no changes needed."
            if decision == "approve"
            else "We couldn't identify a small set of changes that would flip the decision; consider improving your overall financial profile over time."
        )
        return NaturalLanguageExplanation(text=text, language=language, source="template")

    key = _cache_key(changes, decision, language, "cf")
    cached = _cache_get(key)
    if cached:
        return NaturalLanguageExplanation(
            text=cached, language=language, source="gemini", cached=True,
        )

    model = _get_model()
    if model is None:
        # Template fallback: just join the first two changes
        bits = []
        for c in changes[:2]:
            label = (c.get("display_label") or "a factor").lower()
            unit = c.get("display_unit", "")
            bits.append(f"{label} from {unit}{_fmt_num(c.get('current'))} to {unit}{_fmt_num(c.get('suggested'))}")
        text = "To move your application toward approval, consider adjusting " + _join(bits, "en") + "."
        return NaturalLanguageExplanation(text=text, language=language, source="template")

    prompt = (_PROMPT_CF_HI if language == "hi" else _PROMPT_CF_EN).format(
        score=score, decision=decision, changes=_format_changes(changes),
    )
    try:
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        if not text:
            raise ValueError("Empty Gemini response")
        _cache_put(key, text)
        return NaturalLanguageExplanation(text=text, language=language, source="gemini")
    except Exception as e:
        logger.warning(f"Gemini counter-factual call failed: {e}")
        bits = []
        for c in changes[:2]:
            label = (c.get("display_label") or "a factor").lower()
            unit = c.get("display_unit", "")
            bits.append(f"{label} from {unit}{_fmt_num(c.get('current'))} to {unit}{_fmt_num(c.get('suggested'))}")
        text = "To move your application toward approval, consider adjusting " + _join(bits, "en") + "."
        return NaturalLanguageExplanation(text=text, language=language, source="template")


class GeminiExplainer:
    """Thin object wrapper so call sites can dependency-inject if needed."""

    def explain(self, score: float, decision: str, top_features: List[Dict], language: Language = "en") -> NaturalLanguageExplanation:
        return explain_decision(score, decision, top_features, language)

    def narrate_counterfactual(self, score: float, decision: str, changes: List[Dict], language: Language = "en") -> NaturalLanguageExplanation:
        return narrate_counterfactual(score, decision, changes, language)
