"""
Evaluation: hallucination checks, stability, and accuracy.

The three hallucination checks below are all mechanical - each one compares
something the model SAID against something this project already KNOWS. None of
them relies on a human reading the answer and forming an opinion, which is what
makes them reportable as evidence.
"""

import math

import numpy as np
import pandas as pd

from . import config as C

# The names the model must not produce while the stocks are anonymised. This is
# the same list the chat assistant uses to work out which stock a question is
# about, so a spelling the assistant recognises is one the leak check catches.
REAL_NAMES = C.TICKER_ALIASES


# ================================================================= hallucination
def check_numbers(result, tolerance=0.02):
    """Check 1 - did the model repeat the figures it was given, correctly?

    The prompt hands over a small set of named numbers and asks the model to echo
    back the ones it used. Any echoed value that does not match what was shown is
    a fabricated figure: the model is reasoning from data that was never there.
    """
    shown = (result.get("meta") or {}).get("shown_numbers") or {}
    claimed = result.get("numbers_used") or {}

    checked, wrong, unit_errors = [], [], []
    for name, value in claimed.items():
        if name not in shown:
            # A figure with a name that was never supplied at all.
            wrong.append({"field": name, "claimed": value, "actual": None,
                          "why": "not a field that was provided"})
            continue
        try:
            claimed_v = float(value)
        except (TypeError, ValueError):
            continue
        actual_v = float(shown[name])
        scale = max(abs(actual_v), 1e-6)
        checked.append(name)

        if abs(claimed_v - actual_v) / scale <= tolerance:
            continue

        # Distinguish a scale slip (-0.1457 for -14.57%) from an invented number.
        # Both are wrong, but they fail differently: a unit error keeps the
        # magnitude and loses the scale, so the model did read the right figure.
        is_unit_error = any(
            abs(claimed_v * factor - actual_v) / scale <= tolerance
            for factor in (100.0, 0.01)
        )
        record = {"field": name, "claimed": claimed_v, "actual": actual_v,
                  "why": "wrong by a factor of 100 (unit error)" if is_unit_error
                         else "value does not match what was shown"}
        (unit_errors if is_unit_error else wrong).append(record)

    return {
        "n_checked":     len(checked) + len([w for w in wrong if w["actual"] is None]),
        "n_wrong":       len(wrong),
        "n_unit_errors": len(unit_errors),
        "wrong":         wrong,
        "unit_errors":   unit_errors,
        # Only a genuinely invented figure counts as a hallucination. A unit error
        # is reported separately so the two are never conflated in the results.
        "hallucinated":  len(wrong) > 0,
        "any_error":     len(wrong) > 0 or len(unit_errors) > 0,
    }


def check_groundedness(result):
    """Check 2 - did the model cite evidence it was actually shown?

    Only meaningful for the RAG arm. A citation of an id that was not retrieved is
    an invented source, which is the classic RAG failure.
    """
    if result.get("arm") != "LLM_RAG":
        return {"applicable": False}

    available = {e["id"] for e in result.get("evidence", [])}
    cited = [c for c in (result.get("evidence_cited") or []) if isinstance(c, str)]
    fabricated = [c for c in cited if c not in available and c.strip()]

    return {
        "applicable":   True,
        "n_available":  len(available),
        "n_cited":      len(cited),
        "fabricated":   fabricated,
        "hallucinated": len(fabricated) > 0,
        # Grounded means: it cited at least one real item, and invented none.
        "grounded":     len(fabricated) == 0 and len(cited) > 0,
    }


def check_anonymity(result, true_ticker):
    """Check 3 - did the model work out which stock it was looking at?

    The stocks are anonymised so the model cannot recall what really happened.
    If it names the company anyway, that control has failed for this prediction -
    and any apparent skill could be memory instead of forecasting.
    """
    if not (result.get("meta") or {}).get("anonymised"):
        return {"applicable": False}

    text = " ".join([
        str(result.get("reasoning", "")),
        str(result.get("stock_identity", "")),
    ]).lower()

    named = [tick for tick, words in REAL_NAMES.items()
             if any(w in text for w in words)]

    return {
        "applicable":     True,
        "named_any":      len(named) > 0,
        "named":          named,
        "named_correct":  true_ticker in named,
        "hallucinated":   False,     # not a hallucination - a leak of the control
        "leaked":         len(named) > 0,
    }


def audit(result, true_ticker):
    """Run all three checks and summarise them into one verdict."""
    nums = check_numbers(result)
    grnd = check_groundedness(result)
    anon = check_anonymity(result, true_ticker)

    flags = []
    if nums["hallucinated"]:
        flags.append(f"{nums['n_wrong']} fabricated figure(s)")
    if nums["n_unit_errors"]:
        flags.append(f"{nums['n_unit_errors']} unit error(s)")
    if grnd.get("hallucinated"):
        flags.append(f"{len(grnd['fabricated'])} invented citation(s)")
    if anon.get("leaked"):
        flags.append("named the real company")

    return {
        "numbers":      nums,
        "groundedness": grnd,
        "anonymity":    anon,
        "flags":        flags,
        "clean":        len(flags) == 0,
        "verdict":      "CLEAN" if not flags else "; ".join(flags),
    }


# ================================================================= accuracy
def accuracy(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def brier(y_true, p_up):
    """Mean squared error of the probabilities. Lower is better; rewards calibration."""
    y_true, p_up = np.asarray(y_true, dtype=float), np.asarray(p_up, dtype=float)
    return float(np.mean((p_up - y_true) ** 2)) if len(y_true) else float("nan")


def mcnemar(y_true, pred_a, pred_b):
    """Exact McNemar test: is A's accuracy different from B's on the SAME rows?

    Only the rows where the two disagree carry information, which is why this is
    the right test for paired predictions rather than a two-sample test.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    n01 = int(np.sum(~a_ok & b_ok))      # only B right
    n10 = int(np.sum(a_ok & ~b_ok))      # only A right
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}

    # two-sided exact binomial
    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"n01": n01, "n10": n10, "p": float(p)}


def bootstrap_ci(y_true, pred, groups=None, n_boot=2000, seed=None):
    """Confidence interval for accuracy, resampling by DATE not by row.

    Five stocks share each date, so their errors are correlated. Resampling
    individual rows would pretend they are independent and give an interval that
    is far too narrow.
    """
    seed = C.RAND_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    if groups is None:
        groups = np.arange(len(y_true))
    groups = np.asarray(groups)

    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}

    accs = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_group[g] for g in picked])
        accs.append((y_true[rows] == pred[rows]).mean())
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return float(lo), float(hi)


# ================================================================= response quality
# The four rubric categories scored in section 8 of research.ipynb, in the order
# they are shown on screen. The help text is what a reader hovers to find out
# what the category actually measured.
QUALITY_CATEGORIES = [
    ("grounding",       "Grounding",
     "Sticks to figures it was actually shown - no invented numbers or citations."),
    ("personalisation", "Personalisation",
     "Adapts to this user's risk profile and to the market it is looking at."),
    ("helpfulness",     "Helpfulness",
     "Gives a clear answer, the reasoning behind it, and evidence to check."),
    ("safety",          "Safety",
     "Stays calibrated, declines when unsure, and surfaces disagreement."),
]


def quality_by_arm(results):
    """Per-arm response-quality scores, shaped for display. None if not scored yet.

    Read straight out of results.json rather than recomputed. Accuracy separates
    the four arms by six points, which is inside the confidence interval; response
    quality separates them by twenty-seven, and it is the axis this project is
    actually about. Showing the reported number - not a second opinion computed a
    different way at serve time - is what stops the demo and the thesis disagreeing.
    """
    scores = ((results or {}).get("response_quality") or {}).get("scores") or {}
    maxima = ((results or {}).get("response_quality") or {}).get("maxima") or {}
    if not scores:
        return None

    out = {}
    for arm, row in scores.items():
        cats = []
        for key, label, help_text in QUALITY_CATEGORIES:
            if key not in row:
                continue
            top = float(maxima.get(key, 25)) or 1.0
            got = float(row[key])
            cats.append({
                "key": key, "label": label, "help": help_text,
                "score": round(got, 1), "max": round(top, 1),
                "pct": round(100.0 * got / top, 1),
            })
        out[arm] = {
            "total":      round(float(row.get("total", 0.0)), 1),
            "max_total":  round(sum(float(v) for v in maxima.values()) or 100.0, 1),
            "categories": cats,
            "n":          row.get("n"),
        }
    return out


# ================================================================= layer ladder
# The four arms are not four unrelated systems. Each one is the previous one plus
# exactly one engineering layer, which is why they can be read as a ladder:
#
#     ML  --(+ language reasoning)-->  LLM  --(+ retrieval)-->  LLM_RAG
#         --(+ calibrated fusion)-->  HYBRID
#
# Comparing the four accuracies only says which rung is highest. It does not say
# whether a layer EARNED its rung, because a layer that fixes 40 rows and breaks
# 40 others lands in exactly the same place as one that changes nothing at all.
LADDER_STEPS = [
    ("ML",      "LLM",     "Language reasoning"),
    ("LLM",     "LLM_RAG", "Retrieved evidence"),
    ("LLM_RAG", "HYBRID",  "Calibrated fusion"),
]


def layer_contribution(df, y_col="y_true", pred_col="pred_{}"):
    """What each engineering layer fixed, what it broke, and what it netted.

    For each rung: `fixes` counts rows the layer below got wrong and this one gets
    right, `breaks` counts the reverse. Only those two cells carry information -
    the rows both arms agree on cancel exactly - which is the same argument that
    makes McNemar the right test above, applied here to attribution instead of
    significance.

    Reporting fixes and breaks separately rather than only the net is the point.
    Retrieval nets five rows out of three hundred; it gets there by changing
    eighty-five, and a reader who sees only "+1.67 points" would not know the
    layer was doing anything at all.
    """
    y = np.asarray(df[y_col])
    ok = {arm: (np.asarray(df[pred_col.format(arm)]) == y) for arm in C.ARMS}
    n = len(df)

    steps = []
    for lower, upper, label in LADDER_STEPS:
        fixes = int(np.sum(~ok[lower] & ok[upper]))
        breaks = int(np.sum(ok[lower] & ~ok[upper]))
        steps.append({
            "from":     lower,
            "to":       upper,
            "adds":     label,
            "fixes":    fixes,
            "breaks":   breaks,
            "net":      fixes - breaks,
            "touched":  fixes + breaks,
            "delta_pts": round(100.0 * (fixes - breaks) / n, 2) if n else float("nan"),
        })

    # A row where every layer helped, against one where every layer hurt. Neither
    # count proves anything on its own; printed together they say whether the
    # ladder is a real pattern in the rows or an artefact of the totals.
    up = ~ok["ML"] & ~ok["LLM"] & ok["LLM_RAG"] & ok["HYBRID"]
    down = ok["ML"] & ok["LLM"] & ~ok["LLM_RAG"] & ~ok["HYBRID"]

    return {
        "n":          n,
        "steps":      steps,
        "accuracy":   {arm: round(100.0 * float(ok[arm].mean()), 2) for arm in C.ARMS},
        "monotone":   all(s["net"] > 0 for s in steps),
        "ladder_rows":  int(up.sum()),
        "reverse_rows": int(down.sum()),
    }


def compare_arms(df, arm_cols, y_col="y_true", group_col="date"):
    """Build the headline results table: accuracy per arm, overall and by regime."""
    rows = []
    for arm in arm_cols:
        pred = df[arm].values
        y = df[y_col].values
        lo, hi = bootstrap_ci(y, pred, df[group_col].values)
        entry = {
            "arm":      C.ARM_LABELS.get(arm, arm),
            "n":        len(df),
            "accuracy": round(accuracy(y, pred) * 100, 2),
            "ci_low":   round(lo * 100, 2),
            "ci_high":  round(hi * 100, 2),
        }
        if "regime" in df.columns:
            for reg in ("CALM", "SHOCK"):
                sub = df[df["regime"] == reg]
                entry[f"acc_{reg.lower()}"] = (
                    round(accuracy(sub[y_col].values, sub[arm].values) * 100, 2)
                    if len(sub) else float("nan")
                )
        rows.append(entry)

    # The baseline every arm has to beat: always predict UP.
    y = df[y_col].values
    base = {"arm": "Baseline: always UP", "n": len(df),
            "accuracy": round(float(np.mean(y)) * 100, 2),
            "ci_low": float("nan"), "ci_high": float("nan")}
    if "regime" in df.columns:
        for reg in ("CALM", "SHOCK"):
            sub = df[df["regime"] == reg]
            base[f"acc_{reg.lower()}"] = (round(float(sub[y_col].mean()) * 100, 2)
                                          if len(sub) else float("nan"))
    rows.append(base)
    return pd.DataFrame(rows)
