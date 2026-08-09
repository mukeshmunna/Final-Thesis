"""
Arm 4: the Hybrid - calibrated logit pooling.

The fusion is two steps:

    z_arm  = logit(p_arm) - logit(t_arm)          # calibrate
    z_hyb  = (1 - w) * z_ML + w * z_LLM_RAG       # pool
    p_hyb  = sigmoid(z_hyb),  decide UP if p_hyb >= 0.5

Why the calibration step exists
-------------------------------
The two arms do not speak the same language. The LSTM's p(up) sits in a narrow
band around 0.61 and only becomes a real UP call above 0.59; the LLM emits coarse
round numbers (0.15, 0.4, 0.6, 0.75) and only means UP above 0.60. Averaging the
raw probabilities therefore averages two different zero points, and the blend
inherits neither arm's decision boundary. That is exactly what the first version
of this module did, and it is why the hybrid scored below both of its own parents
(53.7%, against 52.7% for the ML arm and 57.7% for the grounded LLM).

Subtracting each arm's own tuned threshold in log-odds fixes this. z is then a
signed distance from that arm's decision point: positive means "this arm says UP,
by this much". Two comparable quantities can be averaged; two probabilities on
different scales cannot. Because both inputs are centred, the pooled score has
its decision point at zero by construction - so the hybrid needs no threshold of
its own, one fitted parameter fewer than every other arm.

Why the weight is shrunk towards equal
--------------------------------------
w is derived per regime from each arm's log-loss on the tuning split, then pulled
halfway back to 0.5. Two reasons, both visible before any holdout number is read:

1. The tuning split holds 200 rows and is 70% shock, while the holdout is 63%
   calm. A weight fitted hard on that sample is fitting its regime mix, not the
   arms. Searching w for accuracy - what the previous version did - drove w to
   0.00 and 0.02, i.e. it threw the LLM away entirely.
2. Log-loss reads the whole probability rather than just which side of the line
   it fell, so it is a far lower-variance estimate of reliability than accuracy
   on a sample this size.

Shrinking towards equal weights is the standard answer to the forecast
combination puzzle: when weights must be estimated from a short, non-stationary
sample, the simple average is hard to beat. Here the derived weights land near
0.49, so the hybrid is close to an equal blend - but it is derived, not asserted,
and it would still move if one arm were genuinely much worse.
"""

import json

import numpy as np

from . import config as C

WEIGHTS_FILE = C.OUTPUT_DIR / "hybrid_weights.json"

# The fusion method this file implements. Written into the weights file so an
# older file, whose numbers meant something different, is never silently reused.
METHOD = "calibrated_logit_pool_v2"

_EPS = 1e-4

# |z| below this is an arm sitting on its own decision boundary: it was asked and
# effectively answered "I do not know". The narration has to treat that as an
# abstention rather than as a vote. Comparing signs alone - which is all the first
# version of this file did - let a card announce "both systems point the same way,
# so this carries the confidence of two independent methods" on a case where one
# of the two read exactly +0.00 and had contributed nothing. That is the one
# sentence a forecasting assistant most needs to get right, because it is the
# sentence a reader uses to decide how much of their own judgement to suspend.
NEUTRAL_BAND = 0.05


# ---------------------------------------------------------------- primitives
def logit(p):
    """Log-odds, clipped so a confidence of exactly 0 or 1 cannot become infinite."""
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def calibrate(p, threshold):
    """How far this arm is from its OWN decision point, in log-odds.

    Positive means the arm is calling UP. Zero means it is exactly on the fence.
    This is the step that makes two differently-scaled arms comparable.
    """
    return logit(p) - logit(threshold)


def pool(z_ml, z_llm, w):
    """Weighted average of two calibrated scores. w is the trust placed in the LLM."""
    return (1.0 - w) * np.asarray(z_ml, dtype=float) + w * np.asarray(z_llm, dtype=float)


def fuse(p_ml, p_llm, w, t_ml=0.5, t_llm=0.5):
    """Full pipeline on probabilities: calibrate, pool, back to a probability."""
    return sigmoid(pool(calibrate(p_ml, t_ml), calibrate(p_llm, t_llm), w))


def log_loss(y_true, p):
    """Mean negative log-likelihood. Lower is better; punishes confident mistakes."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_true, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# ---------------------------------------------------------------- weighting
def reliability_weight(y_true, z_ml, z_llm, shrink=None):
    """Trust in the LLM, from the two arms' log-loss, shrunk towards an equal blend.

    An arm that predicts the tuning rows better gets more of the weight, but the
    shrinkage stops a small sample from handing either arm the whole blend.
    """
    shrink = C.HYBRID_SHRINK if shrink is None else shrink
    ll_ml = log_loss(y_true, sigmoid(z_ml))
    ll_llm = log_loss(y_true, sigmoid(z_llm))
    if ll_ml <= 0 or ll_llm <= 0:
        return 0.5
    raw = (1.0 / ll_llm) / ((1.0 / ll_ml) + (1.0 / ll_llm))
    return round(shrink * 0.5 + (1.0 - shrink) * raw, 3)


def tune_by_regime(tuning_df, thresholds, p_ml_col="p_ml", p_llm_col="p_llm",
                   y_col="y_true", regime_col="regime"):
    """Derive w separately inside each regime. Returns {'CALM': w, 'SHOCK': w}.

    Called with the TUNING rows only. A regime with too few rows to say anything
    falls back to an equal blend rather than to a number fitted on noise.
    """
    t_ml = float(thresholds.get("ML", 0.5))
    t_llm = float(thresholds.get("LLM_RAG", 0.5))

    weights = {}
    for regime in ("CALM", "SHOCK"):
        sub = tuning_df[tuning_df[regime_col] == regime]
        if len(sub) < 20:
            weights[regime] = 0.5
            continue
        weights[regime] = reliability_weight(
            sub[y_col].values,
            calibrate(sub[p_ml_col].values, t_ml),
            calibrate(sub[p_llm_col].values, t_llm),
        )
    return weights


def apply_by_regime(df, weights, thresholds, p_ml_col="p_ml", p_llm_col="p_llm",
                    regime_col="regime"):
    """Apply the derived weights row by row, using each row's own regime."""
    w = df[regime_col].map(weights).fillna(0.5).values
    return fuse(df[p_ml_col].values, df[p_llm_col].values, w,
                t_ml=float(thresholds.get("ML", 0.5)),
                t_llm=float(thresholds.get("LLM_RAG", 0.5)))


# ---------------------------------------------------------------- persistence
def save_weights(weights, thresholds=None):
    payload = {"method": METHOD, "weights": weights, "thresholds": thresholds or {}}
    WEIGHTS_FILE.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def load_weights():
    """Read the derived weights, ignoring any file written by the old fusion.

    An equal blend is the fallback: it is the method's own neutral point, so a
    missing file degrades to "average the two arms" rather than to a guess.
    """
    default = {"CALM": 0.5, "SHOCK": 0.5}
    if not WEIGHTS_FILE.exists():
        return default
    try:
        payload = json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default
    if payload.get("method") != METHOD:
        return default
    return payload.get("weights") or default


def _reading(z):
    """Plain-English reading of one arm's calibrated score.

    The number alone does not tell a reader whether +0.10 is a lot. Naming the
    band next to it does, and it is the band - not the sign - that the verdict
    below is allowed to reason about.
    """
    if abs(z) < NEUTRAL_BAND:
        return "on the fence"
    side = "UP" if z > 0 else "DOWN"
    if abs(z) < 0.30:
        return f"weak {side}"
    if abs(z) < 0.90:
        return f"clear {side}"
    return f"strong {side}"


def _call_strength(p):
    """How much of a call the pooled probability actually is."""
    confidence = p if p >= 0.5 else 1.0 - p
    if confidence < 0.55:
        return "borderline, close to a coin flip"
    if confidence < 0.65:
        return "a lean rather than a conviction"
    if confidence < 0.80:
        return "a moderately confident call"
    return "a confident call"


def load_thresholds():
    """Read the per-arm decision thresholds chosen on the tuning split.

    The web demo uses these too, so a prediction shown on screen is thresholded
    exactly as the reported results were. Without this the demo could call a case
    UP that the thesis counts as DOWN.
    """
    results = C.OUTPUT_DIR / "results.json"
    if results.exists():
        t = json.loads(results.read_text(encoding="utf-8")).get("thresholds") or {}
        if t:
            return t
    return {arm: 0.5 for arm in C.ARMS}


# ---------------------------------------------------------------- serving
class HybridPredictor:
    """Combines one ML result and one LLM result into the hybrid answer."""

    def __init__(self, weights=None, thresholds=None):
        self.weights = weights or load_weights()
        self.thresholds = thresholds or load_thresholds()

    def predict(self, ml_result, llm_result, regime=None):
        if ml_result is None or llm_result is None:
            return None

        regime = regime or ml_result.get("regime") or "CALM"
        w = float(self.weights.get(regime, 0.5))
        t_ml = float(self.thresholds.get("ML", 0.5))
        t_llm = float(self.thresholds.get("LLM_RAG", 0.5))

        z_ml = float(calibrate(ml_result["p_up"], t_ml))
        z_llm = float(calibrate(llm_result["p_up"], t_llm))
        z = float(pool(z_ml, z_llm, w))
        p = float(sigmoid(z))

        # Both inputs are centred on their own decision point, so the pooled
        # score decides at zero - i.e. at p = 0.5. The hybrid is the one arm that
        # needs no threshold fitted for it.
        direction = "UP" if p >= 0.5 else "DOWN"

        confidence = p if p >= 0.5 else 1.0 - p

        # Sign test only, and deliberately left that way: `agree` is read by the
        # front end and by app.py's cross-check note, and it is the same quantity
        # the notebook computes when it scores the arms. The narration needs a
        # stricter test, so that one is derived separately rather than by
        # redefining this one underneath its existing callers.
        agree = (z_ml >= 0) == (z_llm >= 0)

        ml_speaks  = abs(z_ml) >= NEUTRAL_BAND
        llm_speaks = abs(z_llm) >= NEUTRAL_BAND
        confirmed = bool(agree and ml_speaks and llm_speaks)

        if confirmed:
            cross_check = "confirms"
        elif not (ml_speaks or llm_speaks):
            cross_check = "silent"
        elif not llm_speaks:
            cross_check = "llm_abstains"
        elif not ml_speaks:
            cross_check = "ml_abstains"
        else:
            cross_check = "conflicts"

        # The grounded arm's narrative is carried forward, not replaced. The fusion
        # is a check ON that reasoning rather than a substitute for it, so the card
        # should show everything the LLM+RAG arm said PLUS what the cross-check
        # added. Leading with the fusion algebra instead - which is what this used
        # to do - made the most-engineered arm read as the least informative one on
        # screen, because a reader saw "z = -0.33" where the arm below it explained
        # the market in plain English.
        inherited = (llm_result.get("reasoning") or "").strip()

        # The answer first, in the terms the question was asked in: which way, over
        # what horizon, and how much of a call it is. The previous version opened
        # with a sentence about the method and did not state the confidence at all,
        # so a reader had to reach the last line of a single 90-word paragraph
        # before learning that "UP" here meant 51% - and the paragraph's closing
        # sentence encouraged them to read that as strong. Naming the strength in
        # the first line is what stops the prose and the number disagreeing.
        headline = (
            f"Call: {direction} over the next {C.PREDICT_DAYS} trading days, at "
            f"{confidence * 100:.0f}% confidence - {_call_strength(p)}. This is the "
            f"only answer here pooled from two independent systems rather than one."
        )

        market = f"Market read: {inherited}" if inherited else ""

        fusion = (
            f"Cross-check: the same question went independently to the LSTM, which "
            f"reads 60 days of prices and no news. Measured from each system's own "
            f"decision point, the LSTM reads {z_ml:+.2f} ({_reading(z_ml)}) and the "
            f"grounded LLM {z_llm:+.2f} ({_reading(z_llm)}); positive means UP. VIX "
            f"puts this in a {regime.lower()} market, where the derived trust in the "
            f"LLM is {w:.2f}, so the pooled score is {z:+.2f} - {direction}."
        )

        verdict = {
            "confirms": (
                "Verdict: both systems clear their own decision points in the same "
                "direction, so this call rests on two independent methods - one "
                "reading prices, one reading dated evidence - rather than one."),
            "silent": (
                "Verdict: neither system is far enough from its own decision point to "
                "be saying anything. The direction above is which side of the line the "
                "arithmetic fell on, not a signal; read it as no view."),
            "llm_abstains": (
                "Verdict: the grounded LLM is sitting on its own decision boundary, so "
                "it is abstaining, not agreeing. This call rests on the price model "
                "alone - the cross-check ran and came back empty, which is weaker "
                "than two systems agreeing and stronger than never having asked."),
            "ml_abstains": (
                "Verdict: the price model is sitting on its own decision boundary, so "
                "it is abstaining, not agreeing. This call rests on the grounded LLM "
                "alone, with the cross-check adding no support either way."),
            "conflicts": (
                "Verdict: the two systems point opposite ways. The one further from "
                "its own decision point carries the call and the confidence is cut to "
                "say so - either system on its own would have reported this as "
                "certain."),
        }[cross_check]

        # Stated once, at the end, and only where the number warrants it. A caution
        # printed on every answer is read as boilerplate and stops carrying
        # information; printed on the 51% cases only, it marks them out.
        if confidence < 0.55:
            verdict += (
                f" At {confidence * 100:.0f}% the pooled score is barely off the fence, "
                f"and five-day direction is close to unpredictable even when it is not "
                f"- treat this as no clear edge rather than as something to trade on.")

        return {
            "arm":        "HYBRID",
            "p_up":       p,
            "direction":  direction,
            "confidence": confidence,
            "threshold":  0.5,
            "weight_llm": w,
            "weight_ml":  round(1.0 - w, 3),
            "z_ml":       round(z_ml, 3),
            "z_llm":      round(z_llm, 3),
            "agree":      bool(agree),
            # What the cross-check actually did, which is the distinction `agree`
            # cannot draw: an arm on its own boundary has the same sign as one that
            # is genuinely calling UP, and means something completely different.
            "confirmed":   confirmed,
            "cross_check": cross_check,
            "regime":     regime,
            "y_true":     ml_result.get("y_true"),
            # Kept separately so the audit and the notebook can read the fusion on
            # its own, without having to split the displayed prose back apart.
            "fusion_note": fusion,
            "inherited_reasoning": inherited,
            # Blank line between blocks. The card renders this with white-space
            # pre-line, so the four parts - answer, market, cross-check, verdict -
            # arrive as four short paragraphs instead of one 90-word wall that a
            # reader skims and takes the last sentence of.
            "explanation": "\n\n".join(
                x for x in (headline, market, fusion, verdict) if x),
        }


# ---------------------------------------------------------------- personalisation
def apply_profile(prediction, profile_name=None):
    """Turn a prediction into advice for THIS user.

    The risk profile changes only the confidence required before the system will
    suggest acting. It never changes the prediction itself - keeping those two
    separate is what stops personalisation quietly contaminating the evaluation.
    """
    profile_name = profile_name or C.DEFAULT_PROFILE
    profile = C.RISK_PROFILES.get(profile_name, C.RISK_PROFILES[C.DEFAULT_PROFILE])
    threshold = profile["threshold"]
    confidence = float(prediction.get("confidence", 0.0))

    acts = confidence >= threshold
    if acts:
        action = "BUY" if prediction["direction"] == "UP" else "AVOID / SELL"
        reason = (f"Confidence {confidence * 100:.0f}% clears your "
                  f"{profile['label'].lower()} threshold of {threshold * 100:.0f}%.")
    else:
        action = "NO ACTION"
        reason = (f"Confidence {confidence * 100:.0f}% is below your "
                  f"{profile['label'].lower()} threshold of {threshold * 100:.0f}%.")

    return {
        "action":     action,
        "acted":      acts,
        "threshold":  threshold,
        "profile":    profile["label"],
        "reason":     reason,
    }
