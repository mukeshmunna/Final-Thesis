"""
What the assistant knows, and how it decides which kind of answer to give.

The chat window accepts anything, but only one kind of question - "should I buy
X" - can be answered by running the four arms. Everything else used to fall
through to bare evidence retrieval, which answered "I have no evidence for that"
to perfectly reasonable questions like "which model is most accurate?", because
the corpus contains market events, not facts about this project.

This module fixes that by routing the question first:

    forecast     -> run all four arms                        (handled in app.py)
    performance  -> read the real numbers out of results.json
    explain      -> a written answer from FACTS below
    history      -> retrieved market evidence
    capability   -> what this thing can and cannot do

Every non-forecast answer is assembled from either results.json or the FACTS
table, never from the model's own memory. That is the same rule the RAG arm
follows, applied to the assistant itself: if the answer is not in what the system
actually knows, it says so instead of inventing one.
"""

import json
import re

from . import config as C

RESULTS_FILE = C.OUTPUT_DIR / "results.json"


def load_results():
    """The reported numbers, or None if the backtest has not been run."""
    if not RESULTS_FILE.exists():
        return None
    try:
        return json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------- intents
FORECAST_WORDS = (
    "buy", "sell", "hold", "invest", "should i", "worth it", "good buy",
    "forecast", "predict", "prediction", "going up", "go up", "going down",
    "go down", "rise", "fall", "bullish", "bearish", "outlook", "target",
)
PERFORMANCE_WORDS = (
    "accurate", "accuracy", "how good", "how well", "best model", "which model",
    "which system", "best system", "performance", "results", "score", "win rate",
    "better than", "compare", "comparison", "reliable", "trust you",
    "hallucinat", "stability", "stable", "brier", "significant", "baseline",
)
# Questions about the accuracy ceiling get the written answer rather than the
# results table - "why isn't it higher" is asking for a reason, not a number.
CEILING_WORDS = (
    "so low", "too low", "not higher", "should be higher", "beat the market",
    "make money", "low accuracy", "improve accuracy", "get rich", "profitable",
    "only 58", "just 58", "why is it", "is that good", "is this good", "% good",
)
EXPLAIN_WORDS = (
    "what is", "what's", "whats", "what are", "how does", "how do you", "how did",
    "explain", "meaning", "mean by", "define", "why do", "why does", "why is",
    "tell me about", "difference between",
)
CAPABILITY_WORDS = (
    "what can you", "who are you", "what do you do", "help", "hello", "hi ",
    "hey", "capabilities", "how to use", "what should i ask", "options",
)

FORECAST = "forecast"
PERFORMANCE = "performance"
EXPLAIN = "explain"
HISTORY = "history"
CAPABILITY = "capability"


def resolve_ticker(message):
    """Find which stock a question is about, by ticker symbol or company name."""
    low = message.lower()
    for tick, words in C.TICKER_ALIASES.items():
        if any(re.search(rf"\b{re.escape(w)}", low) for w in words):
            return tick
    return None


def classify(message):
    """Decide what kind of question this is.

    A named stock wins outright: "how accurate were you on Tesla" is still best
    answered by running Tesla, with the accuracy figures shown alongside.
    """
    low = f" {message.lower().strip()} "

    if resolve_ticker(message):
        return FORECAST
    if any(w in low for w in CEILING_WORDS):
        return EXPLAIN
    if any(w in low for w in PERFORMANCE_WORDS):
        return PERFORMANCE
    if any(w in low for w in CAPABILITY_WORDS) and len(low.split()) <= 8:
        return CAPABILITY
    if any(w in low for w in EXPLAIN_WORDS) and _explain_topic(message):
        return EXPLAIN
    if any(w in low for w in FORECAST_WORDS):
        return FORECAST          # wants a forecast but named no stock
    if any(w in low for w in EXPLAIN_WORDS):
        return EXPLAIN
    return HISTORY


# ---------------------------------------------------------------- what it knows
# What each measured number means. Five scores get quoted in the performance
# answer, on three different scales, and one of them runs the other way - a lower
# hallucination rate is better while a higher everything-else is. A reader who
# does not already know that reads 5.7% and 58% as though they were comparable,
# and the arm with the *best* auditability looks like the worst one. Printed
# under the numbers as well as answerable on its own, because a definition kept
# one question away only ever reaches the reader who already suspected they
# needed it.
METRIC_GLOSSARY = "\n".join([
    "- **Response quality — out of 100, higher is better.** Four parts: "
    "**grounding** (30) - can every claim be traced to something the system was "
    "actually given; **personalisation** (25) - is the advice shaped to this user's "
    "risk profile and to the market regime, rather than one answer for everyone; "
    "**helpfulness** (25) - is there a clear direction, a reason for it, and "
    "something to check it against; **safety** (20) - is the confidence calibrated, "
    "does it decline when unsure, does it surface disagreement between its own "
    "parts, does it avoid absolutist language. Scored mechanically from saved "
    "responses - no human rater and no LLM judge.",
    "- **Hallucination — a percentage of answers, lower is better.** The share of "
    "responses in which a figure or a citation did not match what the system was "
    "actually shown: a number echoed back that was never in the prompt, or an "
    "evidence id that was never retrieved. It checks claims against the system's "
    "own inputs, which a machine can verify - not against whether the market later "
    "agreed, which it cannot.",
    "- **Groundedness — a percentage of answers, higher is better.** The share that "
    "cited at least one genuinely retrieved evidence item and invented none. Only "
    "the RAG arm can score on it: an answer given no evidence has nothing to be "
    "grounded in, and nothing to be caught out on either, which is exactly why the "
    "ungrounded arm looks cleaner on the hallucination check above.",
    "- **Stability — a percentage of prompts, lower is better.** How often the "
    "identical question, asked again, came back with the opposite direction. The ML "
    "model is a fixed function of its input and scores 0 by construction.",
    "- **Direction accuracy — a percentage of predictions, higher is better,** but "
    "read against the always-UP baseline beside it rather than on its own. Every "
    "confidence interval here overlaps every other one.",
])

# Written answers to the questions this project is actually about. Each entry is
# (trigger words, title, answer). Kept here rather than in the prompt so the
# assistant's explanation of the method cannot drift from the method.
FACTS = [
    # "out of 100" is deliberately NOT a trigger here: it already belongs to the
    # rubric entry below, and _explain_topic breaks ties in list order, so
    # claiming it would silently take that question away from a fuller answer.
    (("what do these numbers mean", "how to read these numbers",
      "read these numbers", "what does the score mean", "glossary", "legend",
      "lower is better", "higher is better"),
     "What each number means",
     METRIC_GLOSSARY),

    (("hybrid", "fusion", "blend", "combine", "pool"),
     "How the hybrid works",
     "The hybrid does not average the two models' probabilities directly - they are "
     "not on the same scale. The LSTM's p(up) sits in a narrow band and only means UP "
     "above its tuned cut-off; the LLM emits coarse round numbers with a different "
     "cut-off. Each arm is first re-expressed as a **signed distance from its own "
     "decision point** in log-odds, so positive always means 'this arm says UP, by "
     "this much'. Those two comparable numbers are then averaged, with the weight "
     "derived per market regime from each arm's log-loss on the tuning split and "
     "shrunk halfway back towards an equal blend. Because both inputs are already "
     "centred, the blend decides at 0.5 and needs no threshold fitted for it - one "
     "free parameter fewer than any arm it is compared against."),

    (("rag", "retriev", "evidence", "grounded", "grounding"),
     "What RAG means here",
     "RAG is retrieval-augmented generation. Before the LLM answers, the system "
     f"searches a corpus of dated market events and hands it the top {C.RAG_TOP_K} "
     f"matches from the {C.RAG_WINDOW_DAYS} days before the question date. Nothing "
     "dated after the question date can ever be retrieved, which is what stops the "
     "model reading the future. Retrieval is plain TF-IDF cosine similarity - no "
     "vector database - so every step can be checked by hand. The LLM must cite the "
     "ids it used, and those citations are checked against what it was actually "
     "shown; a citation to an id that was never retrieved counts as a fabricated "
     "source."),

    (("hallucinat", "made up", "fabricat", "invent"),
     "How hallucination is measured",
     "Three mechanical checks run on every LLM answer, none of which needs a human "
     "to read it. **One:** the model must echo back the figures it used, and each is "
     "compared to what it was actually shown - a mismatch is a fabricated figure, and "
     "an error of exactly a factor of 100 is recorded separately as a unit slip, "
     "because that means it read the right number and lost the scale. **Two:** every "
     "evidence id it cites is checked against the ids it was given. **Three:** the "
     "stock names are hidden behind labels like 'Stock A', so if the answer names the "
     "real company, the anonymity control has failed for that prediction and any "
     "apparent skill could be memory rather than forecasting."),

    (("vix", "regime", "shock", "calm", "volatil"),
     "Calm and shock markets",
     f"A day is labelled SHOCK when the VIX closes above {C.VIX_SHOCK_LEVEL:.0f}, and "
     "CALM otherwise. The VIX is published on the day itself, so labelling a date this "
     "way uses no information from the future. The split matters because the two "
     "regimes are where the models behave differently: the LSTM does respectably in "
     "calm markets and falls to roughly a coin flip in shocks, while the grounded LLM "
     "holds up. Every accuracy figure in this project is reported overall and split "
     "by regime for that reason."),

    (("lstm", "neural", "ml model", "machine learning", "gru", "deep learning"),
     "The ML arm",
     f"An LSTM reading the last {C.SEQ_LEN} trading days of engineered price features "
     f"- returns, RSI, volatility, volume ratio, MACD - and predicting whether the "
     f"close will be higher {C.PREDICT_DAYS} trading days later. It is trained "
     f"walk-forward over {C.N_FOLDS} folds with a {C.EMBARGO_DAYS}-day embargo between "
     "train and test, so no training label overlaps a test window. It is a fixed "
     "function of its input: the same 60 days always produce the same number, which is "
     "why its stability is exactly zero flips."),

    (("threshold", "cut-off", "cutoff", "decision point", "calibrat"),
     "Why each arm has its own threshold",
     "A raw probability of 0.55 does not mean the same thing coming from the LSTM as "
     "it does coming from the LLM. Each arm therefore gets one decision threshold, "
     "chosen on the tuning split only, and every arm gets the same treatment - so no "
     "system is judged at a cut-off another one was allowed to optimise. The demo "
     "applies those same thresholds, which is why a prediction on screen can never "
     "disagree with the thesis. Where the LLM's own wording says UP but the tuned "
     "cut-off makes the decision DOWN, the page shows both."),

    (("risk profile", "cautious", "aggressive", "balanced", "personalis", "personaliz"),
     "What the risk profile changes",
     "Only the confidence required before the system will suggest acting - never the "
     "prediction. Cautious acts at 70%, balanced at 60%, aggressive at 55%; below that "
     "the advice is NO ACTION. Keeping personalisation separate from prediction is "
     "deliberate: if the profile could move the forecast, the evaluation would be "
     "measuring the profile rather than the model."),

    (("stability", "self-consistency", "same question twice", "determinist", "flip"),
     "Stability",
     "The same prompt is sent five times and the disagreements are counted. The LSTM "
     "is deterministic, so it never moves. The LLM is sampled and does sometimes "
     "contradict itself on input it has already seen - and when it does, it is almost "
     "always in a shock period. This is why a single LLM accuracy figure is a sample "
     "rather than a property of the system, and it is measured here rather than "
     "assumed away."),

    (("data", "dataset", "which stocks", "how many", "period", "coverage", "training"),
     "What the system covers",
     f"Five stocks - {', '.join(C.TICKERS)} - over {C.START_DATE} to {C.END_DATE}, "
     f"predicting direction {C.PREDICT_DAYS} trading days ahead. Every date you can "
     "ask about is a past date, which is the point: the true answer is known, so the "
     "page can show you which systems were actually right rather than asking you to "
     "take a forecast on trust."),

    (("so low", "too low", "only 58", "just 58", "not higher", "beat the market",
      "make money", "low accuracy", "bad accuracy", "should be higher",
      "improve accuracy", "get rich", "profitable", "is that good", "is this good",
      "% good"),
     "Why the accuracy is not higher",
     "Because five-day stock direction is close to unpredictable, and this project "
     "reports that rather than hiding it. Look at the baselines on the same rows: "
     "always predicting UP scores 54.3% and a coin flip scores about 52%. Every arm "
     "here sits between 52% and 59%, which is where the published literature on "
     "short-horizon direction forecasting sits too.\n\n"
     "It is worth knowing how hard the ceiling is. If you fit every threshold and "
     "fusion weight *directly on the holdout* - straightforward cheating - the best "
     "achievable blend is 60.3%. Even an oracle that knew, for every single row, "
     "which of the four systems to believe would only reach 82.7%, because on 17% of "
     "rows all four are wrong at the same time.\n\n"
     "So a system reporting 85% on this task would not have found an edge; it would "
     "have a look-ahead leak, or be tuned on its own test set. The contribution here "
     "is not a profitable forecaster - it is a measurement framework: what each "
     "engineering layer adds, and what it costs in hallucination risk and stability."),

    (("response quality", "rubric", "grounding score", "safety score",
      "how are you scored", "answer quality", "100 points", "out of 100"),
     "How the answers themselves are scored",
     "Alongside direction accuracy, all four systems are scored **as assistants** on 100 "
     "points over the same 300 holdout cases: **grounding** (30) - can every claim be "
     "traced to something the system was given; **personalisation** (25) - is the advice "
     "shaped to this user and this situation; **helpfulness** (25) - can the user "
     "understand and check it; and **safety** (20) - is the confidence calibrated, can it "
     "decline to act, does it flag internal disagreement, does it avoid absolutist "
     "language.\n\n"
     "Everything is scored mechanically from saved responses. No human rates anything and "
     "no LLM judges anything - an LLM judge would import the same non-determinism this "
     "project measures elsewhere, and would not reproduce.\n\n"
     "Two caveats that are stated rather than buried: 42 of the 100 points are earned "
     "identically by all four arms, so the differences matter and the totals flatter "
     "everyone; and the text-quality rules are scored on model-generated text only, never "
     "on the fixed fusion sentence, because grading that would measure the project's own "
     "template rather than the system."),

    (("rsi", "macd", "indicator", "feature", "technical"),
     "The price features",
     "The models see engineered features rather than raw prices: RSI(14) on a 0-1 "
     "scale, returns over 5 and 20 days, 10-day volatility, volume against its 20-day "
     "average, and a MACD ratio. The price history shown to the LLM is rebased so the "
     "first day equals 100, which hides the actual price level - another guard against "
     "it recognising the stock and answering from memory."),
]


def _explain_topic(message):
    """The best-matching knowledge entry, or None."""
    low = message.lower()
    best, best_hits = None, 0
    for triggers, title, body in FACTS:
        hits = sum(1 for t in triggers if t in low)
        if hits > best_hits:
            best, best_hits = (title, body), hits
    return best


def explain(message):
    """A written answer about the method, or an honest miss."""
    topic = _explain_topic(message)
    if topic:
        title, body = topic
        return f"**{title}.** {body}"
    return None


# ---------------------------------------------------------------- performance
def performance_answer(results):
    """The headline evaluation numbers, in prose, straight from results.json.

    Deliberately leads with what is measured reliably - auditability, stability,
    calibration - and puts direction accuracy in its place afterwards, with the
    baseline next to it. Quoting "58%" first invites the reader to hear it as a
    good forecaster; it is not, and neither is anything else on this task.
    """
    if not results:
        return ("I have no evaluation results on disk yet - the backtest in section 7 of "
                "research.ipynb has not been run, so I would only be guessing.")

    acc = results.get("accuracy", {})
    hall = results.get("hallucination", {})
    stab = results.get("stability", {})
    n = results.get("n_holdout", 0)
    dates = results.get("n_dates", 0)
    base = acc.get("BASELINE_ALWAYS_UP", {}).get("overall")

    lines = ["Three different things get measured here, and they do not all point the "
             "same way.", ""]

    # ---- 1. auditability
    if hall:
        rag, llm = hall.get("LLM_RAG", {}), hall.get("LLM", {})
        lines += [
            "**Can the AI be caught making things up?** Yes, mechanically, and it "
            f"mostly does not. Across {rag.get('n', 0) + llm.get('n', 0)} responses it "
            f"fabricated a figure **{llm.get('fabricated_number_pct', 0)}%** of the time "
            f"and invented a citation **{llm.get('fabricated_citation_pct', 0)}%** of "
            f"the time. {llm.get('clean_pct')}% of ungrounded and "
            f"{rag.get('clean_pct')}% of grounded answers passed all three checks. "
            "Grounding scores lower because citing evidence creates a way to be caught "
            "citing it wrongly - an ungrounded answer has nothing to check.", "",
        ]

    # ---- 2. stability
    if stab:
        hyb = stab.get("hybrid") or {}
        lines += [
            "**Does it give the same answer twice?** The ML model always does - it is a "
            f"fixed function. The LLM changed its mind on **{stab.get('flip_rate_pct')}%** "
            f"of identical prompts, all of it in shocks "
            f"({stab.get('by_regime', {}).get('SHOCK', 0)}% there against "
            f"{stab.get('by_regime', {}).get('CALM', 0)}% in calm markets).",
        ]
        if hyb:
            lines.append(
                f"The hybrid halves the movement - its probability wobbles "
                f"{hyb.get('p_std_vs_llm')}x as much as the LLM's - but that was **not "
                f"enough to stop it flipping**: it flipped on the same "
                f"{hyb.get('flip_rate_pct')}% of cases. Blending damps the size of the "
                "disagreement, not the fact of it.")
        lines.append("")

    # ---- 3. response quality
    rq = (results.get("response_quality") or {}).get("scores") or {}
    if rq:
        order = sorted(rq, key=lambda a: -rq[a]["total"])
        lines += [
            "**Are the answers themselves any good?** This is the axis where the four "
            "systems genuinely separate. Scored mechanically out of 100 on grounding, "
            "personalisation, helpfulness and safety:", "",
        ]
        for arm in order:
            s = rq[arm]
            lines.append(f"- **{C.ARM_LABELS.get(arm, arm)}** — {s['total']}/100 "
                         f"(grounding {s['grounding']}, personalisation "
                         f"{s['personalisation']}, helpfulness {s['helpfulness']}, "
                         f"safety {s['safety']})")
        top, second = order[0], order[1]
        lines += ["", f"A {round(rq[order[0]]['total'] - rq[order[-1]]['total'], 1)}-point "
                  f"spread, against about six points on accuracy. But the top two are only "
                  f"{round(rq[top]['total'] - rq[second]['total'], 1)} apart, and that is "
                  "structural rather than close-run: the hybrid's answer text *is* the "
                  "grounded LLM's, so it scores identically on grounding and helpfulness. "
                  "It gains on safety and loses on personalisation. **Retrieval is where "
                  "answer quality comes from; fusion adds calibration and the ability to "
                  "say its two halves disagreed.**", ""]

    # ---- 4. accuracy, in context
    ranked = sorted([(k, v["overall"]) for k, v in acc.items() if k in C.ARMS],
                    key=lambda kv: kv[1], reverse=True)
    if ranked:
        best_key, best_acc = ranked[0]
        lines += [
            f"**Does it predict the market?** Barely, and this is the honest part. On "
            f"{n} holdout predictions across {dates} untouched dates:", "",
        ]
        for key, val in ranked:
            e = acc[key]
            ci = e.get("ci")
            lines.append(f"- **{C.ARM_LABELS.get(key, key)}** — {val}%"
                         + (f" (95% CI {ci[0]}–{ci[1]})" if ci else "")
                         + f", calm {e.get('calm')}%, shock {e.get('shock')}%")
        if base is not None:
            lines.append(f"- *Always predict UP* — {base}%, the baseline everything "
                         "has to beat")

        bc = results.get("brier_calibrated", {})
        mc = (results.get("mcnemar") or {}).get("HYBRID_vs_LLM_RAG", {})
        p = mc.get("p")
        lines += ["", f"**{C.ARM_LABELS.get(best_key, best_key)}** is top at {best_acc}%"
                  + (f" and is also the best calibrated ({bc.get('HYBRID')} against "
                     f"{bc.get('LLM_RAG')} for the next best)" if bc else "") + "."]
        if p is not None:
            lines.append(
                f"But every confidence interval here overlaps every other, and against "
                f"the grounded LLM the paired McNemar test gives p = {p:.2f}. On {n} rows "
                f"that gap is noise. **Nobody here beats the market.** Five-day direction "
                f"is close to unpredictable, and a system claiming 85% on it would be "
                f"reporting a data leak, not a discovery. What this project shows is "
                f"which layer adds what, and what each one costs in reliability - not a "
                f"way to trade.")

    lines += ["", metric_glossary()]
    return "\n".join(lines)


# ---------------------------------------------------------------- glossary
def metric_glossary():
    """What each measured number means, printed under the numbers themselves."""
    return "\n".join(["**How to read these numbers**", "", METRIC_GLOSSARY])


# ---------------------------------------------------------------- capability
def capability_answer():
    return (
        "I answer one question with four different AI systems at once, so you can see "
        "what each layer of engineering actually buys.\n\n"
        f"**Ask me about a stock** — {', '.join(C.TICKERS)}, or say Apple, Tesla, "
        "Microsoft, Nvidia or JP Morgan. All four systems answer side by side, and "
        "because these are past dates I also show you which of them were right.\n\n"
        "**Ask me about the method** — how the hybrid combines the models, what RAG is, "
        "how hallucination is measured, why each arm has its own threshold.\n\n"
        "**Ask me how well it works** — the accuracy, calibration and stability numbers "
        "from the holdout evaluation.\n\n"
        "**Ask about a past market period** — I answer from a corpus of dated events, "
        "and only from evidence dated on or before the date in question."
    )


# ---------------------------------------------------------------- fallback
def system_context(results):
    """A compact factual briefing, used to ground any free-form answer."""
    parts = [
        "ABOUT THIS SYSTEM:",
        f"- It compares four forecasting arms on the same question: "
        f"{'; '.join(f'{k} = {v}' for k, v in C.ARM_LABELS.items())}.",
        f"- Coverage: {', '.join(C.TICKERS)}, {C.START_DATE} to {C.END_DATE}, "
        f"predicting direction {C.PREDICT_DAYS} trading days ahead.",
        f"- A market day is SHOCK when VIX > {C.VIX_SHOCK_LEVEL:.0f}, else CALM.",
    ]
    if results:
        acc = results.get("accuracy", {})
        parts.append("- Holdout accuracy: " + ", ".join(
            f"{k} {v['overall']}%" for k, v in acc.items()))
    return "\n".join(parts)
