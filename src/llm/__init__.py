"""LLM-powered explanations for credit decisions.

Wraps SHAP output into natural-language explanations (English/Hindi) via
Gemini Flash, plus narrates counter-factual "how to improve" suggestions.
Falls back gracefully to a template explanation when no API key is configured
or the free-tier quota is exhausted.
"""

from .gemini_explainer import (
    GeminiExplainer,
    NaturalLanguageExplanation,
    explain_decision,
    narrate_counterfactual,
)

__all__ = [
    "GeminiExplainer",
    "NaturalLanguageExplanation",
    "explain_decision",
    "narrate_counterfactual",
]
