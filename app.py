"""
Web demo: a conversational, Hybrid-led assistant backed by four research arms.

    python app.py      ->  http://localhost:5000

Model behavior comes from `src/`, the same engine the research notebook uses.
The presentation layer additionally validates that a five-day future outcome
actually exists before it displays ground truth or direction accuracy.
"""

import csv
import copy
import datetime
import json
import re
import statistics
import traceback

from flask import Flask, jsonify, render_template, request

from src import assistant as A
from src import config as C
from src import data_loader as D
from src import evaluation as V
from src import evidence as E
from src import hybrid as H
from src import llm as L
from src import ml_model as M

app = Flask(__name__)

# Without this the page is cached in memory at start-up, so an edit to the
# template silently does nothing until the whole app is restarted - and start-up
# loads the models, which is slow enough that the edit looks broken instead.
app.config["TEMPLATES_AUTO_RELOAD"] = True

PORT = 5000

# ---------------------------------------------------------------- port check
# Checked here, before the models load, because the failure it prevents is a
# confusing one rather than an obvious one: if another Flask app is still
# running on this port, this one cannot bind, and the browser then shows *that*
# app instead. The page looks wrong rather than absent, which sends you looking
# for a bug in this code. Failing loudly and early is the cheaper outcome.
if __name__ == "__main__":
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _probe:
        if _probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(
                f"\nPort {PORT} is already in use, so the Trading Assistant did NOT start.\n"
                f"Whatever you see at http://127.0.0.1:{PORT} belongs to that other program.\n\n"
                "Stop it first, then run this again:\n"
                f"  Windows   : for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{PORT}') "
                "do taskkill /PID %a /F\n"
                f"  PowerShell: Get-NetTCPConnection -LocalPort {PORT} -State Listen | "
                "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }\n"
                f"  Mac/Linux : lsof -ti tcp:{PORT} | xargs kill -9\n\n"
                "Or change PORT near the top of app.py to serve this app elsewhere.\n"
            )

# ---------------------------------------------------------------- start-up
print("Loading data and models...")
C.set_seeds()

BUNDLE = D.load_everything()

# Ground truth is rebuilt from the price series at start-up instead of trusting
# the saved prediction CSV blindly.  `make_target` deliberately omits dates that
# do not yet have five future trading closes, so a latest-date forecast can never
# be presented as a known DOWN outcome merely because its future value is absent.
TRUTH_BY_KEY = {
    (ticker, index.strftime("%Y-%m-%d")): int(value)
    for ticker, target in BUNDLE["targets"].items()
    for index, value in target.items()
}

try:
    RETRIEVER = E.EvidenceRetriever()
except FileNotFoundError:
    print("  building evidence corpus...")
    E.build_corpus(BUNDLE)
    RETRIEVER = E.EvidenceRetriever()

LLM_ARM = L.LLMArm(BUNDLE, RETRIEVER)
HYBRID  = H.HybridPredictor()

# The same per-arm decision thresholds the reported results were computed with.
# Applying them here is what stops the demo disagreeing with the thesis.
THRESHOLDS = H.load_thresholds()

# The reported response-quality scores, attached to every answer. Accuracy alone
# puts the four arms within six points of each other - inside the confidence
# interval - so a viewer comparing the cards on direction and confidence sees four
# systems that look the same. Quality is where they actually separate, and it is
# already measured; it just was not on screen.
QUALITY = V.quality_by_arm(A.load_results())


def read_json():
    """Parse the request body, returning {} rather than raising on malformed input.

    Without this a bad request surfaces as a 500, which would wrongly look like a
    fault in the model rather than in the caller.
    """
    try:
        return request.get_json(force=True, silent=True) or {}
    except Exception:
        return {}


def apply_threshold(result, arm):
    """Re-decide UP/DOWN using the tuned cut-off for this arm.

    The model's own wording is kept in `stated_direction`, so the page can show
    when the system's decision differs from what the LLM said in prose.
    """
    if result is None:
        return None
    out = dict(result)
    t = float(THRESHOLDS.get(arm, 0.5))
    p = float(out["p_up"])
    out["stated_direction"] = out.get("direction")
    out["direction"]  = "UP" if p >= t else "DOWN"
    out["confidence"] = p if p >= t else 1.0 - p
    out["threshold"]  = t
    return out


try:
    ML = M.MLPredictor()
    ML_READY = True
except FileNotFoundError:
    ML, ML_READY = None, False
    print("  WARNING: no ML predictions yet. Run section 4 of research.ipynb "
          "with RUN_TRAINING = True")


def _saved_backtest_rows():
    """Read the evaluated four-arm rows used by the thesis, if available."""
    path = C.OUTPUT_DIR / "backtest_predictions.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


BACKTEST_ROWS = _saved_backtest_rows()


def _presentation_dates():
    """Dates with saved four-arm answers and a genuinely observable outcome."""
    by_date = {}
    for row in BACKTEST_ROWS:
        key = (row.get("ticker"), row.get("date"))
        if key not in TRUTH_BY_KEY:
            continue
        by_date.setdefault(row["date"], set()).add(row["ticker"])
    dates = sorted(date for date, tickers in by_date.items()
                   if set(C.TICKERS).issubset(tickers))
    if dates or not ML_READY:
        return dates
    # Conservative fallback when the combined table is absent: retain only ML
    # dates for which every stock has a real five-day-ahead target.
    return [date for date in ML.available_dates
            if all((ticker, date) in TRUTH_BY_KEY for ticker in C.TICKERS)]


DEMO_DATES = _presentation_dates()


def _corrected_holdout_record():
    """Direction accuracy after excluding rows whose future close is unavailable."""
    rows = [row for row in BACKTEST_ROWS
            if row.get("split") == "holdout"
            and (row.get("ticker"), row.get("date")) in TRUTH_BY_KEY]
    if not rows:
        return None

    arms = {}
    calm = {}
    shock = {}
    for arm in C.ARMS:
        pred_col = f"pred_{arm}"
        comparable = [row for row in rows if row.get(pred_col) not in (None, "")]
        hits = sum(int(row[pred_col]) == TRUTH_BY_KEY[(row["ticker"], row["date"])]
                   for row in comparable)
        arms[arm] = round(100.0 * hits / len(comparable), 2) if comparable else None

        calm_rows = [row for row in comparable if row.get("regime") == "CALM"]
        calm_hits = sum(
            int(row[pred_col]) == TRUTH_BY_KEY[(row["ticker"], row["date"])]
            for row in calm_rows
        )
        calm[arm] = (round(100.0 * calm_hits / len(calm_rows), 2)
                     if calm_rows else None)

        shock_rows = [row for row in comparable if row.get("regime") == "SHOCK"]
        shock_hits = sum(
            int(row[pred_col]) == TRUTH_BY_KEY[(row["ticker"], row["date"])]
            for row in shock_rows
        )
        shock[arm] = (round(100.0 * shock_hits / len(shock_rows), 2)
                      if shock_rows else None)

    baseline_hits = sum(TRUTH_BY_KEY[(row["ticker"], row["date"])] == 1
                        for row in rows)
    return {
        "n": len(rows),
        "n_dates": len({row["date"] for row in rows}),
        "arms": arms,
        "calm": calm,
        "shock": shock,
        "baseline": round(100.0 * baseline_hits / len(rows), 2),
    }


CORRECTED_HOLDOUT = _corrected_holdout_record()

print(f"Ready. {len(RETRIEVER.items)} evidence items. "
      f"ML predictions: {'yes' if ML_READY else 'NO'}. "
      f"API key: {'yes' if C.GEMINI_KEY else 'NO'}")


def demo_readiness():
    """Describe whether the guided presentation can run without surprises.

    The exact LLM answers used in the evaluation are cached.  Treating a missing
    live key as a broken demo would therefore be inaccurate: the guided examples
    can still run locally, with the same auditable AI outputs used by the thesis.
    This check covers both LLM arms for every curated scenario plus the five saved
    RAG runs for the one stability showcase selected in ``config.py``.
    """
    scenarios = []
    for scenario in C.DEMO_SCENARIOS:
        ticker, date = scenario["ticker"], scenario["date"]
        plain = LLM_ARM.is_cached(ticker, date, use_rag=False)
        rag = LLM_ARM.is_cached(ticker, date, use_rag=True)
        scenarios.append({
            "id": scenario["id"],
            "label": scenario["label"],
            "chat_cached": bool(plain and rag),
        })

    stability = C.STABILITY_DEMO
    stability_cached = all(
        LLM_ARM.is_cached(
            stability["ticker"], stability["date"], use_rag=True, run_index=i
        )
        for i in range(C.SELF_CONSISTENCY_RUNS)
    )
    offline_ready = bool(
        ML_READY
        and scenarios
        and all(s["chat_cached"] for s in scenarios)
        and stability_cached
    )
    live_key_configured = bool(C.GEMINI_KEY)
    if offline_ready and live_key_configured:
        mode = "saved AI responses · live key configured"
    elif offline_ready:
        mode = "saved AI responses"
    elif live_key_configured:
        mode = "needs a guided-demo warm-up"
    else:
        mode = "not ready"

    return {
        # Presentation readiness means the advertised path works without relying
        # on network, quota or key validity.  Key presence is reported separately;
        # it is not proof that a remote call will succeed.
        "ready": offline_ready,
        "offline_ready": offline_ready,
        "live_key_configured": live_key_configured,
        "mode": mode,
        "scenarios": scenarios,
        "stability": {**stability, "cached": bool(stability_cached)},
    }


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    """Everything the front end needs to build its controls."""
    # Only offer dates with saved LLM responses and a real future outcome.  The
    # full ML table is much wider, but most of those combinations would turn a
    # presentation click into an untested live network request.
    dates = DEMO_DATES if ML_READY else []
    readiness = demo_readiness()
    return jsonify({
        "tickers":  C.TICKERS,
        # The spellings the server accepts, sent to the page so "run the stability
        # test on Tesla" can be routed client-side without a round trip and without
        # the page keeping its own second copy of the list.
        "aliases":  C.TICKER_ALIASES,
        "dates":    dates,
        "profiles": [{"key": k, **v} for k, v in C.RISK_PROFILES.items()],
        "arms":     C.ARM_LABELS,
        "ml_ready": ML_READY,
        "has_key":  bool(C.GEMINI_KEY),
        "n_evidence": len(RETRIEVER.items),
        "predict_days": C.PREDICT_DAYS,
        "readiness": readiness,
        "stability_demo": C.STABILITY_DEMO,
        "quality_order": [
            {"key": key, "score": (QUALITY or {}).get(key, {}).get("total")}
            for key in C.ARM_DISPLAY_ORDER
        ],
    })


@app.route("/api/health")
def api_health():
    """Presentation pre-flight: model, evidence and saved/live AI readiness."""
    readiness = demo_readiness()
    payload = {
        "status": "ready" if readiness["ready"] else "not_ready",
        "demo": readiness,
        "components": {
            "ml": {"ready": ML_READY},
            "retrieval": {"ready": bool(RETRIEVER.items),
                          "evidence_items": len(RETRIEVER.items)},
            "llm": {"live_key_configured": bool(C.GEMINI_KEY),
                    "offline_demo": readiness["offline_ready"]},
        },
        "quality_order": [
            {"key": key, "score": (QUALITY or {}).get(key, {}).get("total")}
            for key in C.ARM_DISPLAY_ORDER
        ],
    }
    return jsonify(payload), (200 if readiness["ready"] else 503)


# ---------------------------------------------------------------- stability
@app.route("/api/stability", methods=["POST"])
def api_stability():
    """Send the identical prompt N times and report how often the answer changed.

    This is the demo's answer to 'ML is deterministic, the LLM is not'. Run it
    live in the viva: the ML column never moves, the LLM column sometimes does.
    """
    try:
        body    = read_json()
        ticker  = body.get("ticker", C.TICKERS[0])
        date    = body.get("date")
        use_rag = bool(body.get("use_rag", True))
        runs    = int(body.get("runs", C.SELF_CONSISTENCY_RUNS))

        cons = LLM_ARM.self_consistency(ticker, date, use_rag=use_rag, runs=runs)
        if cons is None:
            return jsonify({"error": "no response from the model"}), 502

        ml = ML.predict(ticker, date) if ML_READY else None

        # What the hybrid would have said on each repeat. Worth showing live: the
        # blend visibly moves less than the LLM does, and yet on a case sitting near
        # the boundary it still flips - which is the honest version of "the hybrid
        # inherits the ML arm's steadiness".
        hyb = None
        if ml is not None:
            regime = ml.get("regime") or D.regime_of(BUNDLE["vix"], date)[0]
            w = float(HYBRID.weights.get(regime, 0.5))
            t_ml = float(THRESHOLDS.get("ML", 0.5))
            t_llm = float(THRESHOLDS.get("LLM_RAG", 0.5))
            fused = [float(H.fuse(ml["p_up"], p, w, t_ml=t_ml, t_llm=t_llm))
                     for p in cons["probs"]]
            dirs = ["UP" if f >= 0.5 else "DOWN" for f in fused]
            majority = max(set(dirs), key=dirs.count)
            hyb = {
                "runs":       len(dirs),
                "directions": dirs,
                "probs":      [round(f, 3) for f in fused],
                "agreement":  round(dirs.count(majority) / len(dirs), 3),
                "flipped":    len(set(dirs)) > 1,
                "p_std":      round(statistics.pstdev(fused) if len(fused) > 1 else 0.0, 4),
                "weight_llm": w,
            }

        return jsonify({
            "hybrid": hyb,
            "ticker": ticker, "date": date, "use_rag": use_rag,
            "llm": {
                "runs":       cons["runs"],
                "directions": cons["directions"],
                "probs":      [round(p, 3) for p in cons["probs"]],
                "agreement":  round(cons["agreement"], 3),
                "flipped":    cons["flipped"],
                "p_mean":     round(cons["p_mean"], 3),
                "p_std":      round(cons["p_std"], 4),
            },
            "ml": {
                # The ML model is a fixed function of its input: same input, same
                # output, every time. Repeating it is pointless, which is the point.
                "runs":      runs,
                "p_up":      round(ml["p_up"], 3) if ml else None,
                "directions": [ml["direction"]] * runs if ml else [],
                "agreement": 1.0,
                "flipped":   False,
                "p_std":     0.0,
            } if ml else None,
        })
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


# ---------------------------------------------------------------- scenarios
@app.route("/api/scenarios")
def api_scenarios():
    """The curated demo cases, used for the suggested-question buttons."""
    return jsonify(C.DEMO_SCENARIOS)


# ---------------------------------------------------------------- chatbot
# Words that mean "as late as the data goes".
_NOW_WORDS = ("now", "today", "currently", "at the moment", "right now", "latest")

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12, "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _month_end(year, month):
    """Last calendar day of a month, as a string."""
    if month == 12:
        nxt = datetime.date(year + 1, 1, 1)
    else:
        nxt = datetime.date(year, month + 1, 1)
    return str(nxt - datetime.timedelta(days=1))


def resolve_date(message, fallback=None):
    """Find the as-of date the question is asking about.

    Recognised, in order of how specific they are: an exact YYYY-MM-DD, a month
    and year ("March 2020"), a bare year, then words meaning "as late as the data
    goes". Anything the retriever is given is used as a hard cut-off - nothing
    dated after it can be read - so parsing "March 2020" properly is not a
    convenience. Without it, a question about March 2020 is answered from evidence
    around whatever date happens to be selected on screen, and the honest reply
    "I have nothing on that" is produced for a period the corpus does cover.

    "Now" resolves to the last date the backtest covers rather than a genuinely
    live date. That is deliberate: it keeps the ground-truth answer available, so
    the demo can show whether each system was actually right.
    """
    low = message.lower()

    m = re.search(r"\d{4}-\d{2}-\d{2}", message)
    if m:
        return m.group(), "the date in your question"

    m = re.search(r"\b(" + "|".join(_MONTHS) + r")\w*\.?\s+(?:of\s+)?(\d{4})\b", low)
    if m:
        year = int(m.group(2))
        return _month_end(year, _MONTHS[m.group(1)]), f"the end of {m.group(1).title()} {year}"

    if DEMO_DATES and any(w in low for w in _NOW_WORDS):
        return DEMO_DATES[-1], "the latest date with a verifiable five-day outcome"

    m = re.search(r"\b(19|20)\d{2}\b", low)
    if m:
        return f"{m.group()}-12-31", f"the end of {m.group()}"

    if fallback:
        return fallback, "the date selected on screen"
    return ((DEMO_DATES[-1] if DEMO_DATES else C.END_DATE),
            "the latest date with a saved, verifiable comparison")


def snap_to_available(date):
    """The latest date the ML model actually has, at or before `date`.

    A question can name any date; the model only has the trading days it was run
    on. Snapping backwards rather than forwards keeps the look-ahead guarantee.

    Returns `(date, how)`: `how` is None when the date was already covered, "back"
    when it moved to an earlier trading day, and "before_range" when the question
    asked about a date earlier than anything the model has. That last case needs
    naming separately rather than folding into "back", because it is the one case
    where the date moves *forward*. The walk-forward holdout starts in July 2021,
    so a question about the COVID crash cannot be answered on its own date - and
    answering it on the earliest date the model does have, while calling that
    "moved back", would file a 2021 answer under a 2020 question.
    """
    if not ML_READY or not DEMO_DATES:
        return date, None
    usable = [d for d in DEMO_DATES if d <= date]
    if not usable:
        return DEMO_DATES[0], "before_range"
    return usable[-1], ("back" if usable[-1] != date else None)


def deanonymise(text, ticker):
    """Put the real ticker back into prose the model wrote about "Stock B".

    The model is shown an anonymous label so it cannot answer from memory of what
    the company actually did. That control belongs in the prompt, not on screen -
    a user who asked about Apple should not be handed an answer about Stock B.

    The audit reads the model's original text, not this one, so restoring the name
    here cannot turn a clean answer into an apparent anonymity leak.
    """
    if not text:
        return text
    label = L.ANON_LABELS.get(ticker)
    if label:
        text = re.sub(rf"\b{re.escape(label)}\b", ticker, text)
    # Any other letter refers to a stock this answer was not about.
    return re.sub(r"\bStock [A-Z]\b", "the stock", text)


def build_arm_list(ml, llm_plain, llm_rag, hyb, ticker, truth_label):
    """Assemble the four answers in presentation order, most-engineered first.

    Returned as an ordered list rather than a dictionary so the front end cannot
    accidentally reorder them - the ranking is part of the explanation.
    """
    raw = {"ML": ml, "LLM": llm_plain, "LLM_RAG": llm_rag, "HYBRID": hyb}
    out = []
    for rank, key in enumerate(C.ARM_DISPLAY_ORDER, start=1):
        res = raw.get(key)
        if res is None:
            continue
        entry = {
            "rank":       rank,
            "key":        key,
            "label":      C.ARM_LABELS[key],
            "tagline":    C.ARM_TAGLINE[key],
            "adds":       C.ARM_WHAT_IT_ADDS[key],
            "direction":  res["direction"],
            "confidence": res["confidence"],
            "p_up":       res["p_up"],
            "threshold":  res.get("threshold"),
            "reasoning":  deanonymise(
                res.get("reasoning") or res.get("explanation") or "", ticker),
            "response_source": (
                ("Saved thesis AI response" if res.get("cached")
                 else "Live Gemini response")
                if key in ("LLM", "LLM_RAG")
                else ("Computed calibrated fusion" if key == "HYBRID"
                      else "Saved walk-forward ML prediction")
            ),
            "shown_as":   (L.ANON_LABELS.get(ticker)
                           if key in ("LLM", "LLM_RAG") and C.ANONYMISE_TICKERS else None),
            "stated_direction": res.get("stated_direction"),
            "weight_llm": res.get("weight_llm"),
            "weight_ml":  res.get("weight_ml"),
            "z_ml":       res.get("z_ml"),
            "z_llm":      res.get("z_llm"),
            "agree":      res.get("agree"),
            "n_evidence": len(res.get("evidence") or []),
            "evidence_cited": res.get("evidence_cited") or [],
            "numbers_used":   res.get("numbers_used") or {},
            # The measured rubric score for this arm, over the whole holdout. Shown
            # on the card so the ranking is visible on every single answer, not only
            # on the one case the viewer happens to have asked about.
            "quality":    (QUALITY or {}).get(key),
        }
        if key in ("LLM", "LLM_RAG"):
            entry["audit"] = V.audit(res, ticker)
        else:
            entry["audit"] = {
                "verdict": ("N/A - not a language model" if key == "ML"
                            else "N/A - arithmetic blend"),
                "clean": True, "flags": [],
            }
        if truth_label:
            entry["correct"] = (res["direction"] == truth_label)
        out.append(entry)
    return out


def current_results():
    """Saved evaluation results with the five unverifiable final rows removed.

    Response-quality and stability measurements are unchanged.  Direction metrics
    are overlaid from the corrected target series so the free-form performance
    answer and the visual scoreboard cannot disagree during a presentation.
    """
    results = copy.deepcopy(A.load_results() or {})
    corrected = CORRECTED_HOLDOUT
    if not results or not corrected:
        return results

    results["n_holdout"] = corrected["n"]
    results["n_dates"] = corrected["n_dates"]
    accuracy = results.setdefault("accuracy", {})
    for arm in C.ARMS:
        row = accuracy.setdefault(arm, {})
        row.update({
            "overall": corrected["arms"].get(arm),
            "calm": corrected["calm"].get(arm),
            "shock": corrected["shock"].get(arm),
            # The saved interval was calculated on 300 rows, so do not attach it
            # to the corrected 295-row estimate.
            "ci": None,
        })
    accuracy.setdefault("BASELINE_ALWAYS_UP", {})["overall"] = corrected["baseline"]

    # These paired/calibration values were also calculated before the target fix.
    # Keeping them beside corrected accuracy would mix two samples.
    results.pop("mcnemar", None)
    results.pop("brier_calibrated", None)
    return results


def track_record():
    """Each arm's holdout accuracy, so a single case is never shown on its own."""
    results = current_results()
    if not results:
        return None
    acc = results.get("accuracy", {})
    corrected = CORRECTED_HOLDOUT or {}
    quality_n = max((row.get("n") or 0 for row in (QUALITY or {}).values()), default=0)
    return {
        "n":      corrected.get("n") or results.get("n_holdout"),
        "n_quality": quality_n or results.get("n_holdout"),
        "arms":   corrected.get("arms") or
                  {k: v.get("overall") for k, v in acc.items() if k in C.ARMS},
        "shock":  corrected.get("shock") or
                  {k: v.get("shock") for k, v in acc.items() if k in C.ARMS},
        "baseline": corrected.get("baseline") or
                    acc.get("BASELINE_ALWAYS_UP", {}).get("overall"),
        # Both axes in one table. Accuracy is the one a reader expects; quality is
        # the one that actually orders the four systems, and putting them side by
        # side is the honest way to say so.
        "quality": {k: v["total"] for k, v in (QUALITY or {}).items() if k in C.ARMS},
        # The older layer-contribution panel is intentionally not served here. Its
        # pairwise transitions are useful, but they are not a controlled causal
        # ablation and should not be presented as proof that every layer caused the
        # gain. The four measured system outcomes are the defensible comparison.
        "ladder": None,
        "order":  C.ARM_DISPLAY_ORDER,
        "adds":   C.ARM_WHAT_IT_ADDS,
    }


def disagreement_note(arms, truth_label=None):
    """Explain how the systems differ without grading a single historical date."""
    dirs = {a["key"]: a["direction"] for a in arms}
    if len(set(dirs.values())) == 1:
        base = "All four systems agree here."
        return base

    note = []
    if dirs.get("LLM") and dirs.get("LLM_RAG") and dirs["LLM"] != dirs["LLM_RAG"]:
        note.append("Retrieval changed the LLM's answer: with evidence it says "
                    f"{dirs['LLM_RAG']}, without it says {dirs['LLM']}.")
    if dirs.get("ML") and dirs.get("LLM_RAG") and dirs["ML"] != dirs["LLM_RAG"]:
        hyb = next((a for a in arms if a["key"] == "HYBRID"), None)
        which = ("the ML model" if hyb and hyb["direction"] == dirs["ML"]
                 else "the grounded LLM")
        note.append("The ML model and the grounded LLM disagree. Each is measured as a "
                    "distance from its own decision point, and the larger distance wins "
                    f"- here that is {which}.")
    return " ".join(note)


def clean_display_name(value):
    """Return a short, presentation-safe name for conversational replies.

    The name personalises only the assistant wording.  It is deliberately kept
    out of the model prompts so a user's identity cannot change the forecast or
    create a new cache key for every presentation attendee.
    """
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = re.sub(r"[^\w .'-]", "", value, flags=re.UNICODE)
    return value[:40].strip()


def build_assistant_response(ticker, date, arms, advice, name=None):
    """Create the one conversational answer shown before the research details.

    This is a deterministic synthesis of outputs already produced by the four
    evaluated systems, not a fifth model.  In particular, it never reads the
    retrospective ground-truth label.  That keeps the displayed recommendation
    a forecast and lets the saved thesis responses remain fast and reproducible.
    """
    primary = next((arm for arm in arms if arm["key"] == "HYBRID"), None)
    grounded = next((arm for arm in arms if arm["key"] == "LLM_RAG"), None)
    primary = primary or grounded or next((arm for arm in arms if arm.get("reasoning")), arms[0])
    grounded = grounded or primary

    profile = (advice or {}).get("profile", "Balanced")
    action = (advice or {}).get("action", "NO ACTION")
    direction = primary.get("direction")
    direction_word = "upward" if direction == "UP" else "downward"
    display_name = clean_display_name(name)
    salutation = f"{display_name}, " if display_name else ""

    if action == "BUY":
        lead = (f"{salutation}the Hybrid analysis supports a buy for {ticker} over "
                f"this five-trading-day view.")
        decision = (f"The combined evidence leans {direction_word} and is strong enough "
                    f"for your {profile.lower()} preference.")
    elif action == "AVOID / SELL":
        lead = (f"{salutation}the Hybrid analysis suggests avoiding or reducing {ticker} "
                f"over this five-trading-day view.")
        decision = (f"The combined evidence leans {direction_word} and is strong enough "
                    f"for your {profile.lower()} preference.")
    else:
        lead = (f"{salutation}the Hybrid analysis does not support taking action on "
                f"{ticker} right now.")
        decision = (f"The combined evidence leans {direction_word}, but it is not strong "
                    f"enough for your {profile.lower()} preference.")

    # No name means no salutation to carry the opening capital, and the lead is
    # the first sentence the viewer reads.
    if not salutation:
        lead = lead[:1].upper() + lead[1:]

    matching = sum(arm.get("direction") == direction for arm in arms)
    if matching == len(arms):
        cross_check = "All four research systems point in the same direction on this snapshot."
    elif primary.get("agree") is False:
        cross_check = ("The price-pattern model and the evidence-grounded language model "
                       "disagree, so the Hybrid answer treats the result cautiously.")
    else:
        cross_check = (f"{matching} of the {len(arms)} research systems support the "
                       "Hybrid direction on this snapshot.")

    source = grounded.get("response_source") or "Research system response"
    if source == "Saved thesis AI response":
        source_note = ("Uses a saved Gemini response from the thesis evaluation, combined "
                       "with the local ML cross-check.")
    elif source == "Live Gemini response":
        source_note = "Uses a live Gemini response, combined with the local ML cross-check."
    else:
        source_note = "Built from the four local research-system outputs."

    return {
        "lead": lead,
        "decision": decision,
        "rationale": grounded.get("reasoning") or primary.get("reasoning") or "",
        "cross_check": cross_check,
        "source": source_note,
        "basis_arm": primary.get("key"),
        "action": action,
        "profile": profile,
        "ticker": ticker,
        "date": date,
    }


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """The assistant. Routes the question, then answers it from what it knows.

    A question naming a stock runs the full four-arm pipeline. Every other kind
    of question is answered from either the saved evaluation results or the
    system's own factual notes - never from the model's memory. That is the same
    grounding rule the RAG arm follows, applied to the assistant itself.
    """
    try:
        body    = read_json()
        message = (body.get("message") or "").strip()
        profile = body.get("profile", C.DEFAULT_PROFILE)
        name = clean_display_name(body.get("name"))
        if not message:
            return jsonify({"error": "message is required"}), 400

        ticker = body.get("ticker") or A.resolve_ticker(message)
        date, date_source = resolve_date(message, body.get("date"))
        intent = A.FORECAST if ticker else A.classify(message)

        # ---- questions about the system itself, answered from real numbers
        if intent == A.PERFORMANCE:
            return jsonify({
                "type":   "information",
                "topic":  "How well it works",
                "date":   date,
                "answer": A.performance_answer(current_results()),
                "hint":   "Ask about a stock to see all four systems answer the same question.",
            })

        if intent == A.CAPABILITY:
            return jsonify({
                "type":   "information",
                "topic":  "What I can do",
                "date":   date,
                "answer": A.capability_answer(),
            })

        if intent == A.EXPLAIN:
            written = A.explain(message)
            if written:
                return jsonify({
                    "type":   "information",
                    "topic":  "How it works",
                    "date":   date,
                    "answer": written,
                })
            # No note covers it - fall through to the grounded free-form answer.

        # ---- a forecast was wanted but no stock was named
        if intent == A.FORECAST and ticker is None:
            return jsonify({
                "type":   "information",
                "topic":  "Which stock?",
                "date":   date,
                "answer": ("I can only forecast the five stocks this project was built "
                           "and evaluated on, so I need you to name one rather than "
                           "guess. Say **Apple, Tesla, Microsoft, Nvidia or JP Morgan** "
                           f"(or {', '.join(C.TICKERS)}) and all four systems will "
                           "answer the same question side by side."),
            })

        # ---- anything else: answer strictly from retrieved evidence
        if ticker is None:
            evidence = RETRIEVER.retrieve(message, date)
            return jsonify({
                "type":     "information",
                "topic":    "From the evidence corpus",
                "date":     date,
                "answer":   _evidence_answer(message, evidence, date),
                "evidence": evidence,
                "hint":     ("Name one of " + ", ".join(C.TICKERS) +
                             " - or say Apple, Tesla, Microsoft, Nvidia or JP Morgan - "
                             "to get a forecast from all four systems."),
            })

        # ---- run all four arms on the same question.
        #      The date is first snapped back to a day the model actually covers,
        #      so "should I buy Apple in March 2020" is answered on the last
        #      trading day of March 2020 rather than failing on a weekend.
        date, moved = snap_to_available(date)
        if moved == "back":
            date_source += ", moved back to the nearest saved comparison date"
        elif moved == "before_range":
            date_source = ("the earliest saved comparison - you asked about an earlier "
                           "date, so this answer is not about the date you named")

        ml = ML.predict(ticker, date) if ML_READY else None
        if ml is None and ML_READY:
            ml = ML.predict_live(BUNDLE, ticker, date)
        regime = (ml or {}).get("regime") or D.regime_of(BUNDLE["vix"], date)[0]

        llm_plain = LLM_ARM.predict(ticker, date, use_rag=False)
        llm_rag   = LLM_ARM.predict(ticker, date, use_rag=True)
        if any(result is not None and not result.get("cached")
               for result in (llm_plain, llm_rag)):
            # Batch evaluation flushes once at the end. A web request has no such
            # end-of-batch hook, so persist any newly warmed presentation answer
            # before the process exits.
            L.flush_cache()
        hyb       = HYBRID.predict(ml, llm_rag, regime=regime) if (ml and llm_rag) else None

        ml        = apply_threshold(ml, "ML")
        llm_plain = apply_threshold(llm_plain, "LLM")
        llm_rag   = apply_threshold(llm_rag, "LLM_RAG")

        # Recompute the label from the fixed target series.  Saved ML CSVs created
        # before the target fix can contain a false DOWN label on their final rows.
        y_true = TRUTH_BY_KEY.get((ticker, date))
        truth = ({"y_true": int(y_true), "label": "UP" if y_true else "DOWN"}
                 if y_true is not None else None)
        truth_label = truth["label"] if truth else None

        arms = build_arm_list(ml, llm_plain, llm_rag, hyb, ticker, truth_label)
        if not arms:
            return jsonify({"error": "no system could answer for that date"}), 502

        headline = hyb or llm_rag or ml
        advice = H.apply_profile(headline, profile) if headline else None

        return jsonify({
            "type":        "prediction",
            "ticker":      ticker,
            "date":        date,
            "date_source": date_source,
            "regime":      regime,
            "vix":         (ml or {}).get("vix"),
            "horizon":     C.PREDICT_DAYS,
            "answer":      (f"All four systems answered your question about **{ticker}** "
                            f"as of **{date}** ({date_source}), looking "
                            f"{C.PREDICT_DAYS} trading days ahead."),
            "arms":        arms,
            "advice":      advice,
            "truth":       truth,
            "evidence":    (llm_rag or {}).get("evidence") or [],
            "note":        disagreement_note(arms, truth_label),
            "weights":     HYBRID.weights,
            "assistant_response": build_assistant_response(
                ticker, date, arms, advice, name=name),
            # One case proves nothing on its own. The holdout record travels with
            # every answer so the page can always put the two side by side.
            "track_record": track_record(),
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


def _evidence_answer(message, evidence, date=None):
    """Answer a general question from retrieved evidence and the system's own facts.

    The briefing about the system is included because a question the corpus cannot
    answer is often one the project itself can - "what does the hybrid do" is not
    a market event. Both sources are named in the prompt and the model is told to
    use nothing else, so an answer that is not supported by one of them should
    come back as "I do not know" rather than as an invention.
    """
    if not C.GEMINI_KEY:
        if not evidence:
            return ("I have no evidence in my corpus that matches that question, and no "
                    "API key is set, so I am not going to guess. Try naming one of the "
                    "five stocks, or ask how the system works.")
        bullets = "\n".join(f"- ({e['date']}) {e['text']}" for e in evidence)
        return f"Here is the evidence I retrieved:\n\n{bullets}"

    ev = ("\n".join(f"[{e['id']}] ({e['date']}) {e['text']}" for e in evidence)
          if evidence else "(nothing in the corpus matched this question)")

    prompt = (
        "You are the assistant for a stock-forecasting research system. Answer the "
        "user's question using ONLY the two sources below: the system briefing and "
        "the retrieved market evidence. If neither covers it, say plainly that you "
        "do not know rather than guessing. Never give financial advice. Be concise "
        "- at most three short paragraphs - and plain-spoken.\n\n"
        f"{A.system_context(current_results())}\n\n"
        f"RETRIEVED MARKET EVIDENCE (nothing dated after {date or 'the question date'}):\n"
        f"{ev}\n\n"
        f"QUESTION: {message}\n\n"
        'Reply as JSON: {"answer": "..."}'
    )
    raw, _ = L.call_gemini(prompt)
    try:
        return json.loads(raw).get("answer", raw)
    except json.JSONDecodeError:
        return raw


if __name__ == "__main__":
    print(f"Trading Assistant  ->  http://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False)
