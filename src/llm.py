"""
Arms 2 and 3: the LLM, without and with retrieved evidence.

The only difference between the two arms is whether an EVIDENCE section is
present in the prompt. Everything else - model, generation policy, wording and
output format - is identical, so any difference in the results is attributable
to the retrieval step and nothing else.

Two things this module is built to expose
-----------------------------------------
1. Instability. The same prompt is sent several times so that the model's
   disagreement with itself can be counted (see `self_consistency`).
2. Hallucination. The model must report the numbers it used and the evidence it
   relied on, which lets `evaluation.py` check both against the truth.
"""

import hashlib
import json
import os
import re
import threading
import time

import pandas as pd

from . import config as C
from .data_loader import regime_of

# Anonymous labels prevent a model from substituting remembered company history
# for the controlled market snapshot it is asked to evaluate.
ANON_LABELS = {t: f"Stock {chr(65 + i)}" for i, t in enumerate(C.TICKERS)}


class LLMServiceError(RuntimeError):
    """A safe, actionable description of an upstream Gemini failure."""

    def __init__(self, message, *, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = bool(retryable)


def _generation_family(model=None):
    """Return the request-policy family for a supported model identifier."""
    model_name = (model or C.GEMINI_MODEL).rsplit("/", 1)[-1].lower()
    if model_name.startswith("gemini-2.5"):
        return "gemini-2.5"
    if model_name in {
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-pro-latest",
    }:
        return "gemini-3+"
    match = re.match(r"gemini-(\d+)(?:\.|-)", model_name)
    if match and int(match.group(1)) >= 3:
        return "gemini-3+"
    return "unsupported"


def _uses_legacy_generation_config(model=None):
    """Gemini 2.5 uses token budgets; Gemini 3+ uses thinking levels."""
    return _generation_family(model) == "gemini-2.5"


def _is_evaluated_model(model=None):
    configured = (model or C.GEMINI_MODEL).rsplit("/", 1)[-1].lower()
    evaluated = C.EVALUATED_GEMINI_MODEL.rsplit("/", 1)[-1].lower()
    return configured == evaluated


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
                    loaded = json.loads(C.LLM_CACHE.read_text(encoding="utf-8"))
                    _CACHE = loaded if isinstance(loaded, dict) else {}
                except (OSError, json.JSONDecodeError):
                    _CACHE = {}
            else:
                _CACHE = {}
        return _CACHE


def flush_cache():
    """Atomically write the cache so interruption cannot leave partial JSON."""
    with _CACHE_LOCK:
        if _CACHE is not None:
            temporary = C.LLM_CACHE.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(_CACHE, indent=1), encoding="utf-8")
            temporary.replace(C.LLM_CACHE)


def _legacy_cache_key(prompt, run_index):
    """Key used by the original Gemini 2.5 cache, retained for thesis replays."""
    h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    return f"{h}_run{run_index}"


def _cache_key(prompt, run_index):
    """Separate responses by prompt, run, model and generation policy."""
    family = _generation_family()
    if family == "gemini-2.5":
        generation = f"temperature={C.LLM_TEMPERATURE}|thinking_budget=0"
    elif family == "gemini-3+":
        generation = f"thinking_level={C.GEMINI_THINKING_LEVEL}"
    else:
        generation = "unsupported"
    policy = f"{C.GEMINI_MODEL}|{generation}|max_tokens={C.LLM_MAX_TOKENS}|{prompt}"
    h = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:20]
    return f"v2_{h}_run{run_index}"


def _cached_text(cache, prompt, run_index):
    """Return current-model data, with legacy replay only when explicitly enabled."""
    keys = [_cache_key(prompt, run_index)]
    if _is_evaluated_model() or C.GEMINI_USE_LEGACY_CACHE:
        keys.append(_legacy_cache_key(prompt, run_index))
    for key in keys:
        value = cache.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def has_cached_response(prompt, run_index=0):
    """Return whether this exact model request can run without the network.

    The presentation uses previously generated, auditable Gemini responses.  A
    lightweight readiness check lets the UI distinguish "no live API key" from
    "the guided demo cannot run"; those are not the same thing when the exact
    response is already in the research cache.
    """
    if not prompt:
        return False
    cache = _load_cache()
    with _CACHE_LOCK:
        return _cached_text(cache, prompt, run_index) is not None


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
def _generation_config(types):
    """Build a request compatible with the configured Gemini model family."""
    options = {
        "max_output_tokens": C.LLM_MAX_TOKENS,
        "response_mime_type": "application/json",
    }
    family = _generation_family()
    if family == "gemini-2.5":
        options.update({
            "temperature": C.LLM_TEMPERATURE,
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        })
    elif family == "gemini-3+":
        options["thinking_config"] = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel(C.GEMINI_THINKING_LEVEL)
        )
    else:
        raise LLMServiceError(
            f"Gemini model '{C.GEMINI_MODEL}' is not supported by this application. "
            "Use a Gemini 3 model (recommended: gemini-3.6-flash).",
            retryable=False,
        )
    return types.GenerateContentConfig(**options)


def _service_error(exc):
    """Convert a provider exception into a sanitized, retry-aware error."""
    status = getattr(exc, "code", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None

    retryable = bool(
        status in {408, 409, 425, 429}
        or (status is not None and 500 <= status <= 599)
    )
    if status == 400:
        message = (
            f"Gemini rejected the request for model '{C.GEMINI_MODEL}'. "
            "Check GEMINI_MODEL and restart the application."
        )
    elif status in {401, 403}:
        message = "Gemini authentication failed. Check GEMINI_API_KEY and restart the application."
    elif status == 404:
        message = (
            f"Gemini model '{C.GEMINI_MODEL}' is unavailable for this API key. "
            "Set GEMINI_MODEL to a supported model and restart the application."
        )
    elif status == 429:
        message = "Gemini's rate limit or quota was reached. Try again later."
    elif status is not None and status >= 500:
        message = "Gemini is temporarily unavailable. Try again later."
    else:
        message = "The Gemini request failed. Try again later."
    return LLMServiceError(message, status_code=status, retryable=retryable)


def _passes_validator(text, validator):
    if validator is None:
        return True
    try:
        return bool(validator(text))
    except Exception:
        return False


def call_gemini(
    prompt, run_index=0, use_cache=True, max_retries=3, validator=None
):
    """Send one prompt. Returns the raw text.

    Raises when the key is missing rather than returning anything invented, so a
    result in this project can never rest on a fabricated response.
    """
    key = _cache_key(prompt, run_index)
    if use_cache:
        cache = _load_cache()
        with _CACHE_LOCK:
            cached = _cached_text(cache, prompt, run_index)
            if cached is not None and _passes_validator(cached, validator):
                return cached, True                    # (text, was_cached)

    try:
        C.require_key()
    except RuntimeError as exc:
        raise LLMServiceError(
            "GEMINI_API_KEY is not configured. Add it to .env and restart the application.",
            retryable=False,
        ) from exc

    from google import genai
    from google.genai import errors, types

    transport_error_types = [TimeoutError, ConnectionError]
    try:
        import httpx
        transport_error_types.append(httpx.TransportError)
    except ImportError:
        pass
    try:
        import requests
        transport_error_types.append(requests.exceptions.RequestException)
    except ImportError:
        pass
    transport_errors = tuple(transport_error_types)

    client = genai.Client(api_key=C.GEMINI_KEY)
    attempts = max(1, int(max_retries))
    config = _generation_config(types)
    for attempt in range(attempts):
        try:
            resp = client.models.generate_content(
                model=C.GEMINI_MODEL,
                contents=prompt,
                config=config,
            )
            text = (resp.text or "").strip()
            if not text or not _passes_validator(text, validator):
                if attempt == attempts - 1:
                    detail = "an empty response" if not text else "JSON in an unexpected format"
                    raise LLMServiceError(
                        f"Gemini returned {detail}. Try again later.",
                        retryable=True,
                    )
                time.sleep(2 ** attempt)
                continue
            if use_cache:
                cache = _load_cache()
                with _CACHE_LOCK:
                    cache[key] = text
            return text, False
        except errors.APIError as exc:
            error = _service_error(exc)
            if not error.retryable or attempt == attempts - 1:
                raise error from exc
            time.sleep(2 ** attempt)
        except transport_errors as exc:
            if attempt == attempts - 1:
                raise LLMServiceError(
                    "The Gemini service could not be reached. Try again later.",
                    retryable=True,
                ) from exc
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

        raw, cached = call_gemini(
            prompt,
            run_index=run_index,
            use_cache=use_cache,
            validator=lambda text: parse_response(text) is not None,
        )
        parsed = parse_response(raw)
        if parsed is None:
            raise LLMServiceError(
                "Gemini returned JSON that does not match the forecast schema. Try again later.",
                retryable=True,
            )

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
