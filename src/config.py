"""
Every setting for the project, in one file.

Both the research notebook and the web demo import from here, so a number can
never drift between "what the thesis reports" and "what the demo shows".
"""

import os
import random
from pathlib import Path

# ---------------------------------------------------------------- reproducibility
RAND_SEED = 42


def set_seeds():
    """Fix every random source. Called once at start-up by the notebook and the app."""
    random.seed(RAND_SEED)
    os.environ["PYTHONHASHSEED"] = str(RAND_SEED)
    try:
        import numpy as np
        np.random.seed(RAND_SEED)
    except ImportError:
        pass


# ---------------------------------------------------------------- folders
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_DIR     = BASE_DIR / "data"
MODEL_DIR    = BASE_DIR / "saved_models"
OUTPUT_DIR   = BASE_DIR / "outputs"
CACHE_DIR    = DATA_DIR / "cache"

# Load local overrides before any environment-backed setting is declared.  The
# file is git-ignored; importing config therefore works the same way from the
# notebook, the Flask app and one-off command-line checks.
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

for _d in (DATA_DIR, MODEL_DIR, OUTPUT_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

EVIDENCE_FILE = DATA_DIR / "evidence_corpus.json"
LLM_CACHE     = CACHE_DIR / "llm_cache.json"

# ---------------------------------------------------------------- experiment design
TICKERS      = ["NVDA", "AAPL", "TSLA", "JPM", "MSFT"]
START_DATE   = "2019-01-01"
END_DATE     = "2024-12-31"

PREDICT_DAYS = 5     # predict direction 5 trading days ahead
SEQ_LEN      = 60    # the LSTM reads the last 60 days
N_FOLDS      = 3     # walk-forward folds
EMBARGO_DAYS = 5     # gap between train and test, so no label overlaps the boundary
VAL_FRAC     = 0.15  # last slice of each training window, for early stopping

# ---------------------------------------------------------------- regime definition
# A "shock" day is simply VIX > 25. The VIX is published on the day itself, so
# labelling a date this way uses no information from the future.
VIX_TICKER      = "^VIX"
VIX_SHOCK_LEVEL = 25.0

# ---------------------------------------------------------------- the four arms
# These are the names used everywhere: in the results tables and in the UI.
ARMS = ["ML", "LLM", "LLM_RAG", "HYBRID"]

ARM_LABELS = {
    "ML":      "ML only (LSTM)",
    "LLM":     "LLM only (no evidence)",
    "LLM_RAG": "LLM + RAG (grounded)",
    "HYBRID":  "Hybrid (ML + RAG-LLM)",
}

# The order the assistant presents its answers in: most-engineered system first,
# simplest last. Each step down the list removes one capability, so a reader can
# see what each layer contributed.
ARM_DISPLAY_ORDER = ["HYBRID", "LLM_RAG", "LLM", "ML"]

ARM_TAGLINE = {
    "HYBRID":  "Calibrates the ML model and the grounded LLM to their own decision points, "
               "then pools them by market regime",
    "LLM_RAG": "The LLM, but it must retrieve dated evidence before answering",
    "LLM":     "The LLM alone - no outside information, nothing to check it against",
    "ML":      "The LSTM alone - 60 days of prices, no idea what is happening in the world",
}

ARM_WHAT_IT_ADDS = {
    "HYBRID":  "adds calibrated fusion",
    "LLM_RAG": "adds retrieved evidence",
    "LLM":     "adds language reasoning",
    "ML":      "the baseline - patterns only",
}

# ---------------------------------------------------------------- demo scenarios
# Curated cases for the demo, each verified against the holdout predictions.
# They are chosen to show a different behaviour, not to flatter the system - one
# of the four has the Hybrid getting the answer wrong, and it is kept for that
# reason.
DEMO_SCENARIOS = [
    {
        "id":       "grounding",
        "label":    "Retrieval changes the decision",
        "question": "Should I buy MSFT?",
        "ticker":   "MSFT",
        "date":     "2022-08-26",
        "context":  "Jackson Hole - the Fed chair's hawkish speech; markets fell hard.",
        "shows":    ("The LSTM and the ungrounded LLM both say UP. Giving the same LLM dated "
                     "evidence changes its call to DOWN, and the calibrated hybrid follows the "
                     "grounded signal. This case illustrates the controlled retrieval effect; "
                     "the full holdout comparison, not one date, supports the overall ranking."),
    },
    {
        "id":       "ml_wins",
        "label":    "The LLM is too pessimistic",
        "question": "Should I buy AAPL?",
        "ticker":   "AAPL",
        "date":     "2022-10-11",
        "context":  "Mid-crisis, just before a hot inflation print.",
        "shows":    ("Both LLM arms read the bad news and say DOWN, and the hybrid follows them. "
                     "The market rose, so only the LSTM was right. This is the LLM's "
                     "characteristic failure - it reasons like a person reading headlines, but "
                     "public news is already in the price - and it is also the hybrid's limit: "
                     "when both of its inputs lean the same wrong way, fusion cannot rescue it."),
    },
    {
        "id":       "ml_fails",
        "label":    "The ML model fails in a shock",
        "question": "Should I buy NVDA?",
        "ticker":   "NVDA",
        "date":     "2022-08-31",
        "context":  "High volatility - VIX well above the stress threshold.",
        "shows":    ("The LSTM confidently says UP and is wrong. Both LLM arms say DOWN and are "
                     "right, and the hybrid goes with them because the LLM's distance from its "
                     "own decision point is the larger of the two. This is the headline finding: "
                     "the ML model breaks down exactly where the LLM holds up."),
    },
    {
        "id":       "agreement",
        "label":    "A calm day - everyone agrees",
        "question": "Should I buy MSFT?",
        "ticker":   "MSFT",
        "date":     "2022-11-18",
        "context":  "Calm market, VIX below the stress threshold.",
        "shows":    ("All four systems agree and all four are right. In calm conditions "
                     "nothing meaningfully separates them - which is itself a result, and "
                     "the reason the analysis is split by regime."),
    },
]

# The five cached runs for this shock-period case contain an actual LLM direction
# flip (UP, UP, DOWN, DOWN, DOWN).  The UI uses it as the stability showcase so a
# presentation never depends on five fresh network calls or on whichever date the
# viewer happened to select previously.
STABILITY_DEMO = {
    "ticker": "TSLA",
    "date": "2022-05-06",
    "label": "Shock-period stability showcase",
}

# Plain-English names the assistant should understand in a question.
TICKER_ALIASES = {
    "NVDA": ["nvda", "nvidia"],
    "AAPL": ["aapl", "apple"],
    "TSLA": ["tsla", "tesla"],
    "JPM":  ["jpm", "jpmorgan", "jp morgan", "morgan chase", "jp-morgan"],
    "MSFT": ["msft", "microsoft"],
}

# ---------------------------------------------------------------- LLM settings
# Gemini 2.5 Flash is no longer available to every API account.  Use Google's
# current stable Flash model for live calls, while allowing an explicit override
# for reproducibility or a future migration without another source-code change.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"
EVALUATED_GEMINI_MODEL = "gemini-2.5-flash"

# Gemini 3 uses thinking levels instead of the Gemini 2.5 thinking-token budget.
# "minimal" is the closest supported setting to the no-thinking policy used by
# the original experiment, and keeps interactive calls quick and inexpensive.
GEMINI_THINKING_LEVEL = (
    os.getenv("GEMINI_THINKING_LEVEL", "minimal").strip().lower() or "minimal"
)
if GEMINI_THINKING_LEVEL not in {"minimal", "low", "medium", "high"}:
    raise ValueError(
        "GEMINI_THINKING_LEVEL must be one of: minimal, low, medium, high"
    )

# Old prompt-only cache keys contain responses from the evaluated Gemini 2.5
# experiment. Never mix them into current-model calls unless a user explicitly
# opts into thesis replay (or configures the original 2.5 model itself).
GEMINI_USE_LEGACY_CACHE = (
    os.getenv("GEMINI_USE_LEGACY_CACHE", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

# Retained for replaying the original Gemini 2.5 experiment. Modern Gemini
# models deprecate sampling-temperature overrides, so live Gemini 3 calls do not
# send this value.
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS  = 2048

# How many times the same prompt is sent when measuring stability.
# 5 is enough to detect an unstable answer without multiplying the API bill by more.
SELF_CONSISTENCY_RUNS = 5

# Hide the real stock name from the LLM. Showing "NVDA" could test memorised
# company history rather than forecasting from the controlled prompt.
ANONYMISE_TICKERS = True

# ---------------------------------------------------------------- RAG settings
# Retrieval is TF-IDF cosine similarity - deliberately simple, so every step can
# be explained and checked by hand. No vector database, no embeddings.
RAG_TOP_K       = 4     # how many evidence items the LLM is shown
RAG_WINDOW_DAYS = 90    # only evidence dated at or before the query date, within 90 days
                        # (<= query date is what prevents look-ahead)

# ---------------------------------------------------------------- hybrid fusion
# The hybrid pools two CALIBRATED scores rather than two raw probabilities:
#     z_arm   = logit(p_arm) - logit(threshold_arm)     # distance from its own call
#     z_hyb   = (1 - w) * z_ml + w * z_llm
#     p_hyb   = sigmoid(z_hyb),  decide UP at 0.5
# Calibrating first is what lets two differently-scaled arms be averaged at all;
# see the header of src/hybrid.py for why the raw-probability blend failed.
WEIGHT_GRID = [round(x * 0.02, 2) for x in range(51)]   # 0.00 .. 1.00, for the figure

# w is derived per regime from each arm's log-loss on the tuning split, then
# pulled this far back towards an equal blend. 0.0 = trust the tuning split
# completely, 1.0 = ignore it and always average. 0.5 is the compromise: the
# tuning split is only 200 rows and its regime mix does not match the holdout's,
# so a weight fitted hard on it fits the sample rather than the arms.
HYBRID_SHRINK = 0.5

# ---------------------------------------------------------------- personalisation
# The risk profile changes the confidence the system demands before it will act.
# It does NOT change the prediction itself - only whether the user is told to act.
# Keeping those two things separate is what makes the evaluation honest.
RISK_PROFILES = {
    "cautious": {
        "label":       "Cautious",
        "threshold":   0.70,   # act only when at least 70% confident
        "description": "Only acts on high-confidence signals. Most predictions become NO ACTION.",
    },
    "balanced": {
        "label":       "Balanced",
        "threshold":   0.60,
        "description": "Acts on moderately confident signals.",
    },
    "aggressive": {
        "label":       "Aggressive",
        "threshold":   0.55,
        "description": "Acts on any signal above a coin flip by a small margin.",
    },
}
DEFAULT_PROFILE = "balanced"

# ---------------------------------------------------------------- API key
# The key is read from the environment, never written in the code.
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")


def require_key():
    """Raise a clear error if the key is missing.

    This deliberately raises instead of returning a fake answer, so that no
    result in this project can ever be based on invented data.
    """
    if not GEMINI_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Copy .env.example to .env and paste your key from "
            "https://aistudio.google.com/app/apikey\n"
            "Nothing is faked when the key is missing - the run stops here instead."
        )
    return GEMINI_KEY
