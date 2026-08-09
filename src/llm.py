"""
Arms 2 and 3: the LLM, without and with retrieved evidence.

The only difference between the two arms is whether an EVIDENCE section is
present in the prompt. Everything else - model, temperature, wording, output
format - is identical, so any difference in the results is attributable to the
retrieval step and nothing else.

Two things this module is built to expose
-----------------------------------------
1. Instability. The same prompt is sent several times so that the model's
   disagreement with itself can be counted (see `self_consistency`).
2. Hallucination. The model must report the numbers it used and the evidence it
   relied on, which lets `evaluation.py` check both against the truth.
"""

import hashlib
import json
import re
import threading
import time

import pandas as pd

from . import config as C
from .data_loader import regime_of

# Anonymous labels. Gemini's training data covers 2019-2024, so showing "NVDA"
# would test what it remembers, not what it can forecast.
ANON_LABELS = {t: f"Stock {chr(65 + i)}" for i, t in enumerate(C.TICKERS)}


# ---------------------------------------------------------------- cache
# Held in memory and guarded by a lock, because the backtest calls the API from
# several threads at once. Every response is kept, so a re-run costs nothing and
# the reported numbers can be reproduced without paying for the API again.
_CACHE_LOCK = threading.Lock()
_CACHE = None


def _load_cache():
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            if C.LLM_CACHE.exists():
                try:
                    _CACHE = json.loads(C.LLM_CACHE.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    _CACHE = {}
            else:
                _CACHE = {}
        return _CACHE


def flush_cache():
    """Write the cache to disk. Called at the end of a batch, not per response."""
    with _CACHE_LOCK:
        if _CACHE is not None:
            C.LLM_CACHE.write_text(json.dumps(_CACHE, indent=1), encoding="utf-8")


def _cache_key(prompt, run_index):
    """One entry per (prompt, run number), so repeated runs are cached separately."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{h}_run{run_index}"


def has_cached_response(prompt, run_index=0):
    """Return whether this exact model request can run without the network.

    The presentation uses previously generated, auditable Gemini responses.  A
    lightweight readiness check lets the UI distinguish "no live API key" from
    "the guided demo cannot run"; those are not the same thing when the exact
    response is already in the research cache.
    """
    if not prompt:
        return False
    key = _cache_key(prompt, run_index)
    cache = _load_cache()
    with _CACHE_LOCK:
        return key in cache


# ---------------------------------------------------------------- prompt building
def _price_block(bundle, ticker, date, anonymise=True):
    """The market facts shown to the model, plus the same numbers in machine form.

    Returning both matters: the second value is the list of figures the model was
    actually given, which is what the hallucination check compares against.
    """
    label = ANON_LABELS[ticker] if anonymise else ticker
    feats = bundle["features"][ticker]
    prices = bundle["prices"][ticker]

    prior = feats.index[feats.index <= pd.Timestamp(date)]
    if len(prior) == 0:
        return None, None
    row = feats.loc[prior[-1]]

    hist = prices.loc[:pd.Timestamp(date), "Close"].astype(float)
    closes = hist.tail(20).values
    rebased = closes / closes[0] * 100          # starts at 100, so the real price is hidden

    shown = {
        "rsi_14":     round(float(row["rsi_14"]), 3),
        "ret_5d_pct": round(float(row["ret_5d"]) * 100, 2),
        "ret_20d_pct": round(float(row["ret_20d"]) * 100, 2),
        "vol_10d":    round(float(row["vol_10d"]), 4),
        "vol_ratio":  round(float(row["vol_ratio"]), 2),
        "macd_ratio": round(float(row["macd_ratio"]), 5),
    }

    text = (
        f"{label} (price history rebased so the first day = 100):\n"
        f"  Last 20 closes: {', '.join(f'{p:.1f}' for p in rebased)}\n"
        f"  RSI(14), 0-1 scale: {shown['rsi_14']}\n"
        f"  Return over last 5 days: {shown['ret_5d_pct']:+.2f}%\n"
        f"  Return over last 20 days: {shown['ret_20d_pct']:+.2f}%\n"
        f"  10-day volatility: {shown['vol_10d']}\n"
        f"  Volume vs its 20-day average: {shown['vol_ratio']}\n"
        f"  MACD ratio: {shown['macd_ratio']}"
    )
    return text, shown


def build_prompt(bundle, ticker, date, evidence=None, anonymise=True):
    """Build the prompt for one stock on one date.

    `evidence=None`  -> the LLM-only arm.
    `evidence=[...]` -> the LLM+RAG arm; the retrieved items are inserted verbatim.
    """
    block, shown = _price_block(bundle, ticker, date, anonymise)
    if block is None:
        return None, None

    regime, vix_level = regime_of(bundle["vix"], date)

    parts = [
        "You are a quantitative analyst. Predict whether this stock's closing price "
        f"will be HIGHER in {C.PREDICT_DAYS} trading days than it is now.",
        "",
        "MARKET DATA (everything below is as of today; you cannot see the future)",
        "",
        block,
    ]

    if evidence is not None:
        if evidence:
            ev_lines = [
                f"  [{e['id']}] ({e['date']}) {e['text']}  -- source: {e['source']}"
                for e in evidence
            ]
            parts += [
                "",
                "RETRIEVED EVIDENCE (dated on or before today; this is the only outside "
                "information you have):",
                *ev_lines,
                "",
                "Base any statement about market conditions ONLY on the evidence above. "
                "If the evidence does not cover something, say so rather than assuming it.",
            ]
        else:
            parts += [
                "",
                "RETRIEVED EVIDENCE: none found for this date.",
                "",
                "You have no outside information. Judge on the price data alone and say so.",
            ]

    parts += [
        "",
        "Reply with JSON only, in exactly this form:",
        json.dumps({
            "direction": "up or down",
            "confidence": "integer 0-100",
            "reasoning": "two sentences at most",
            "numbers_used": {"rsi_14": 0.0, "ret_5d_pct": 0.0},
            "evidence_ids": ["the id of each evidence item you relied on, [] if none"],
            "stock_identity": "the real company name if you can tell, otherwise 'unknown'",
        }, indent=1),
        "",
        "numbers_used must repeat back the exact figures you based your answer on, "
        "copied from the data above. Do not invent figures that are not shown.",
    ]

    meta = {
        "shown_numbers": shown,
        "regime":        regime,
        "vix":           float(vix_level),
        "evidence_ids":  [e["id"] for e in (evidence or [])],
        "anonymised":    anonymise,
    }
    return "\n".join(parts), meta


# ---------------------------------------------------------------- calling
def call_gemini(prompt, run_index=0, use_cache=True, max_retries=3):
    """Send one prompt. Returns the raw text.

    Raises when the key is missing rather than returning anything invented, so a
    result in this project can never rest on a fabricated response.
    """
    key = _cache_key(prompt, run_index)
    if use_cache:
        cache = _load_cache()
        with _CACHE_LOCK:
            if key in cache:
                return cache[key], True                # (text, was_cached)

    C.require_key()
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=C.GEMINI_KEY)
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(
                model=C.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=C.LLM_TEMPERATURE,
                    max_output_tokens=C.LLM_MAX_TOKENS,
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            text = (resp.text or "").strip()
            if use_cache:
                cache = _load_cache()
                with _CACHE_LOCK:
                    cache[key] = text
            return text, False
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)


def parse_response(text):
    """Pull the JSON out of a reply. Returns None if it cannot be read."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if not m:
            return None
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict) or "direction" not in obj:
        return None

    direction = str(obj.get("direction", "")).strip().lower()
    if direction.startswith("u"):
        direction = "UP"
    elif direction.startswith("d"):
        direction = "DOWN"
    else:
        # A malformed direction is not evidence for DOWN.  Reject it so callers
        # can surface the model failure instead of silently creating a forecast.
        return None
    try:
        conf = float(obj.get("confidence", 50))
    except (TypeError, ValueError):
        conf = 50.0
    conf = max(0.0, min(100.0, conf))

    return {
        "direction":      direction,
        "confidence":     conf,
        "reasoning":      str(obj.get("reasoning", ""))[:600],
        "numbers_used":   obj.get("numbers_used") if isinstance(obj.get("numbers_used"), dict) else {},
        "evidence_ids":   obj.get("evidence_ids") if isinstance(obj.get("evidence_ids"), list) else [],
        "stock_identity": str(obj.get("stock_identity", "unknown"))[:120],
    }


# ---------------------------------------------------------------- the arms
class LLMArm:
    """Runs the LLM with or without retrieved evidence."""

    def __init__(self, bundle, retriever=None):
        self.bundle = bundle
        self.retriever = retriever

    def is_cached(self, ticker, date, use_rag=False, run_index=0):
        """Whether one arm/date is available as a saved, offline AI response."""
        evidence = None
        if use_rag:
            if self.retriever is None:
                return False
            from .evidence import market_state_query
            query = market_state_query(self.bundle, ticker, date)
            evidence = self.retriever.retrieve(query, date)

        prompt, _ = build_prompt(
            self.bundle,
            ticker,
            date,
            evidence=evidence,
            anonymise=C.ANONYMISE_TICKERS,
        )
        return has_cached_response(prompt, run_index=run_index)

    def predict(self, ticker, date, use_rag=False, run_index=0, use_cache=True):
        """One prediction. Set `use_rag=True` for the grounded arm."""
        evidence = None
        if use_rag:
            if self.retriever is None:
                raise ValueError("use_rag=True needs a retriever")
            from .evidence import market_state_query
            query = market_state_query(self.bundle, ticker, date)
            evidence = self.retriever.retrieve(query, date)

        prompt, meta = build_prompt(self.bundle, ticker, date,
                                    evidence=evidence,
                                    anonymise=C.ANONYMISE_TICKERS)
        if prompt is None:
            return None

        raw, cached = call_gemini(prompt, run_index=run_index, use_cache=use_cache)
        parsed = parse_response(raw)
        if parsed is None:
            return None

        # p_up is a probability, so the two arms can be compared with the ML model
        # and blended in the hybrid.
        p_up = parsed["confidence"] / 100.0
        if parsed["direction"] == "DOWN":
            p_up = 1.0 - p_up

        return {
            "arm":            "LLM_RAG" if use_rag else "LLM",
            "p_up":           p_up,
            "direction":      parsed["direction"],
            "confidence":     parsed["confidence"] / 100.0,
            "reasoning":      parsed["reasoning"],
            "numbers_used":   parsed["numbers_used"],
            "evidence_cited": parsed["evidence_ids"],
            "stock_identity": parsed["stock_identity"],
            "evidence":       evidence or [],
            "meta":           meta,
            "cached":         cached,
            "run_index":      run_index,
        }

    def self_consistency(self, ticker, date, use_rag=False, runs=None):
        """Send the identical prompt several times and measure the disagreement.

        This is the answer to 'you cannot compare a deterministic model with a
        non-deterministic one'. Instability is not assumed away - it is measured,
        and reported as a property of the arm.
        """
        runs = runs or C.SELF_CONSISTENCY_RUNS
        outs = []
        for i in range(runs):
            r = self.predict(ticker, date, use_rag=use_rag, run_index=i)
            if r is not None:
                outs.append(r)
        if not outs:
            return None

        dirs  = [o["direction"] for o in outs]
        probs = [o["p_up"] for o in outs]
        majority = max(set(dirs), key=dirs.count)

        return {
            "runs":            len(outs),
            "directions":      dirs,
            "probs":           probs,
            "majority":        majority,
            "agreement":       dirs.count(majority) / len(dirs),   # 1.0 = perfectly stable
            "flipped":         len(set(dirs)) > 1,                 # did it contradict itself?
            "p_mean":          sum(probs) / len(probs),
            "p_std":           float(pd.Series(probs).std(ddof=0)),
            "outputs":         outs,
        }
