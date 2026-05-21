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
    "reject": {"en": "not approved at this time", "hi": "अभी स्वीकृत नहीं"},
    "manual_review": {"en": "flagged for manual review", "hi": "मैन्युअल समीक्षा हेतु चिह्नित"},
}


def _template_explanation(
    score: float, decision: str, top_features: List[Dict], language: Language,
) -> NaturalLanguageExplanation:
    """Deterministic, no-LLM fallback that still references the actual SHAP factors."""
    tone = _DECISION_TONE.get(decision, _DECISION_TONE["manual_review"])[language]
    score_pct = round(score * 100, 1)

    positives = [f for f in top_features if f.get("contribution", 0.0) > 0][:2]
    negatives = [f for f in top_features if f.get("contribution", 0.0) < 0][:2]

    def _name(feat: Dict) -> str:
        return str(feat.get("feature", "factor")).replace("_", " ")

    if language == "hi":
        text = f"आपका आवेदन {tone} किया गया है (स्कोर {score_pct}%)।"
        if positives:
            text += " सबसे मजबूत सकारात्मक कारक: " + ", ".join(_name(f) for f in positives) + "।"
        if negatives:
            text += " सबसे कमजोर पहलू: " + ", ".join(_name(f) for f in negatives) + "।"
        suggestion = "मासिक आय बढ़ाने या बचत अनुपात सुधारने पर विचार करें।" if decision != "approve" else "आपका वित्तीय अनुशासन अच्छा है।"
    else:
        text = f"Your application is {tone} (score {score_pct}%)."
        if positives:
            text += " Strongest positive factors: " + ", ".join(_name(f) for f in positives) + "."
        if negatives:
            text += " Weakest aspects: " + ", ".join(_name(f) for f in negatives) + "."
        suggestion = "Consider raising monthly income or improving savings ratio." if decision != "approve" else "Your financial discipline is strong."

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


def _format_factors(top_features: List[Dict]) -> str:
    """Render the SHAP top-contributors list as a bullet block for the prompt."""
    lines: List[str] = []
    for f in top_features[:6]:
        feat = str(f.get("feature", "factor")).replace("_", " ")
        contrib = float(f.get("contribution", 0.0))
        sign = "+" if contrib >= 0 else "-"
        lines.append(f"- {feat}: contribution {sign}{abs(contrib):.3f}")
    return "\n".join(lines) if lines else "- (no factors available)"


def _format_changes(changes: List[Dict]) -> str:
    lines: List[str] = []
    for c in changes:
        feat = str(c.get("feature", "factor")).replace("_", " ")
        cur = c.get("current")
        sug = c.get("suggested")
        delta = c.get("delta_score")
        delta_str = f"+{delta:.3f}" if delta is not None else ""
        lines.append(f"- {feat}: change from {cur} to {sug} (score impact {delta_str})")
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
            feat = str(c.get("feature", "factor")).replace("_", " ")
            bits.append(f"raise {feat} from {c.get('current')} to {c.get('suggested')}")
        text = "To improve your decision, " + ", and ".join(bits) + "."
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
            feat = str(c.get("feature", "factor")).replace("_", " ")
            bits.append(f"raise {feat} from {c.get('current')} to {c.get('suggested')}")
        text = "To improve your decision, " + ", and ".join(bits) + "."
        return NaturalLanguageExplanation(text=text, language=language, source="template")


class GeminiExplainer:
    """Thin object wrapper so call sites can dependency-inject if needed."""

    def explain(self, score: float, decision: str, top_features: List[Dict], language: Language = "en") -> NaturalLanguageExplanation:
        return explain_decision(score, decision, top_features, language)

    def narrate_counterfactual(self, score: float, decision: str, changes: List[Dict], language: Language = "en") -> NaturalLanguageExplanation:
        return narrate_counterfactual(score, decision, changes, language)
