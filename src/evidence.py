"""
The retrieval half of the RAG arm.

Why this exists
---------------
In the first version of this project I wrote the market context for each date by
hand. That is a real weakness: I chose what the LLM was told, so I shaped its
answer. Here the LLM is given nothing by default. It must RETRIEVE evidence from
a fixed, dated corpus, and the choice is made by an algorithm, not by me.

Two things follow from that, and both are measurable:
  1. Retrieval can be scored (did it fetch the relevant item?).
  2. Generation can be audited (did the answer stay inside the evidence?).

How retrieval works
-------------------
TF-IDF cosine similarity, with a recency weight. Deliberately simple: no vector
database and no embeddings, so every step can be checked by hand and explained
in one sentence.

The no-look-ahead rule
----------------------
An evidence item is only visible if its date is <= the query date. That single
filter is what stops the LLM reading tomorrow's newspaper.
"""

import json
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config as C


# ================================================================= the corpus
# Documented macro events. Each one records only what was KNOWN on its date -
# never what happened next. Sources are named so a reader can verify them.
MACRO_EVENTS = [
    ("2019-08-01", "US announces a new round of tariffs on Chinese imports, widening the trade dispute.", "US Trade Representative"),
    ("2019-08-14", "The 2-year/10-year Treasury yield curve inverts for the first time since 2007, a signal markets read as a recession warning.", "US Treasury yield data"),
    ("2019-10-30", "The Federal Reserve cuts the target rate to 1.50-1.75%, its third cut of the year.", "Federal Reserve FOMC statement"),
    ("2020-01-30", "The World Health Organization declares the coronavirus outbreak a Public Health Emergency of International Concern.", "WHO statement"),
    ("2020-02-24", "Equity markets fall sharply as coronavirus cases spread beyond China; volatility rises quickly.", "Market data"),
    ("2020-03-11", "The World Health Organization declares COVID-19 a pandemic.", "WHO statement"),
    ("2020-03-15", "The Federal Reserve cuts rates to near zero and restarts large-scale asset purchases in an emergency weekend action.", "Federal Reserve FOMC statement"),
    ("2020-03-16", "The VIX closes at 82.69, its highest close on record. Trading halts are triggered on major US indices.", "CBOE data"),
    ("2020-03-23", "The Federal Reserve announces open-ended asset purchases and new credit facilities to support market functioning.", "Federal Reserve statement"),
    ("2020-11-03", "United States presidential election. The outcome is not called on the night, leaving markets uncertain.", "Public record"),
    ("2020-11-09", "Pfizer and BioNTech report high efficacy in their COVID-19 vaccine trial, prompting a large rotation between market sectors.", "Company announcement"),
    ("2021-01-06", "Disruption at the US Capitol during the certification of the election result.", "Public record"),
    ("2021-11-30", "The Federal Reserve chair signals that the word 'transitory' is no longer the right description of inflation, opening the door to faster tightening.", "Congressional testimony"),
    ("2022-01-03", "US equity indices reach record highs as markets begin to price in interest-rate rises.", "Market data"),
    ("2022-02-24", "Russia begins a full-scale invasion of Ukraine. Energy and commodity prices rise sharply.", "Public record"),
    ("2022-03-16", "The Federal Reserve raises rates by 0.25%, the first increase of the tightening cycle.", "Federal Reserve FOMC statement"),
    ("2022-06-15", "The Federal Reserve raises rates by 0.75%, its largest single increase since 1994, after a higher-than-expected inflation reading.", "Federal Reserve FOMC statement"),
    ("2022-09-13", "US inflation data comes in above expectations, triggering one of the largest single-day equity falls of the year.", "US Bureau of Labor Statistics"),
    ("2022-10-13", "US inflation data is released amid heavy market volatility; equities swing sharply within the session.", "US Bureau of Labor Statistics"),
    ("2023-03-10", "Silicon Valley Bank is closed by regulators, the largest US bank failure since 2008. Concern spreads to other regional banks.", "FDIC statement"),
    ("2023-03-19", "UBS agrees to acquire Credit Suisse in a transaction arranged by Swiss authorities.", "Swiss National Bank statement"),
    ("2023-05-24", "A major semiconductor company issues revenue guidance far above expectations, citing demand for AI data-centre hardware.", "Company earnings release"),
    ("2023-07-26", "The Federal Reserve raises rates to 5.25-5.50%, which proves to be the final increase of the cycle.", "Federal Reserve FOMC statement"),
    ("2024-08-05", "A rapid unwinding of yen-funded carry trades causes a sharp global equity sell-off and a spike in volatility.", "Market data"),
    ("2024-11-05", "United States presidential election.", "Public record"),
]


def _fmt_date(d):
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def build_corpus(bundle, save=True):
    """Build the evidence corpus and (optionally) write it to data/.

    Three kinds of item, in increasing order of objectivity:
      macro   - documented world events (the list above)
      vix     - automatically generated when the volatility regime changes
      move    - automatically generated on unusually large market days

    The last two are computed straight from the price data, so most of the
    corpus involves no judgement from me at all.
    """
    items = []

    # ---- 1. documented macro events
    for date, text, source in MACRO_EVENTS:
        items.append({
            "id":     f"macro_{date}",
            "date":   date,
            "type":   "macro",
            "text":   text,
            "source": source,
        })

    # ---- 2. volatility regime changes, generated from the VIX itself
    vix = bundle["vix"]
    above = vix > C.VIX_SHOCK_LEVEL
    crossings = above.ne(above.shift(1)) & above.notna()
    for date in vix.index[crossings]:
        if pd.Timestamp(date) < pd.Timestamp(C.START_DATE):
            continue
        level = float(vix.loc[date])
        state = "rises above" if level > C.VIX_SHOCK_LEVEL else "falls back below"
        items.append({
            "id":     f"vix_{_fmt_date(date)}",
            "date":   _fmt_date(date),
            "type":   "vix",
            "text":   (f"Market volatility {state} the stress threshold: the VIX closes at "
                       f"{level:.1f} against a threshold of {C.VIX_SHOCK_LEVEL:.0f}."),
            "source": "CBOE VIX daily close",
        })

    # ---- 3. unusually large moves in the 5-stock basket
    mkt_ret = bundle["mkt_close"].pct_change()
    big = mkt_ret[mkt_ret.abs() > 0.03]           # more than 3% in a single day
    for date, ret in big.items():
        direction = "rises" if ret > 0 else "falls"
        items.append({
            "id":     f"move_{_fmt_date(date)}",
            "date":   _fmt_date(date),
            "type":   "move",
            "text":   (f"The equal-weighted basket of the five stocks {direction} "
                       f"{abs(ret) * 100:.1f}% in one session, an unusually large daily move."),
            "source": "Computed from daily closing prices",
        })

    items.sort(key=lambda x: x["date"])

    if save:
        C.EVIDENCE_FILE.write_text(json.dumps(items, indent=1), encoding="utf-8")

    return items


def load_corpus():
    """Read the corpus from disk. Raises if it has not been built yet."""
    if not C.EVIDENCE_FILE.exists():
        raise FileNotFoundError(
            f"No evidence corpus at {C.EVIDENCE_FILE}. Run build_corpus() first."
        )
    return json.loads(C.EVIDENCE_FILE.read_text(encoding="utf-8"))


# ================================================================= the retriever
class EvidenceRetriever:
    """TF-IDF retrieval over the evidence corpus, with a date cut-off.

    Scoring is:   final = cosine_similarity * recency_weight

    where recency_weight decays linearly from 1.0 on the query date to 0.3 at the
    edge of the window. Without it, a vivid but old event (COVID, say) would win
    every query for years afterwards.
    """

    def __init__(self, items=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.items = items if items is not None else load_corpus()
        self.dates = np.array([pd.Timestamp(it["date"]) for it in self.items])

        # The date is indexed in words as well as digits ("March 2020" and
        # "2020-03-11"), so a question phrased around a date can match. Without
        # this, "what happened in March 2020?" scores zero against every item.
        corpus_text = []
        for it, d in zip(self.items, self.dates):
            corpus_text.append(
                f"{it['type']} {d.strftime('%B %Y %B %Y')} {it['date']} {it['text']}")

        # min_df=1 because the corpus is small; stop_words removes 'the', 'a', etc.
        self.vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        self.matrix = self.vectorizer.fit_transform(corpus_text)

    def retrieve(self, query, as_of, top_k=None, window_days=None):
        """Return the top-k evidence items known on or before `as_of`.

        `query`  - free text. In the backtest it is built from market state; in
                   the chatbot it is whatever the user typed.
        `as_of`  - the cut-off date. Nothing later than this is ever visible.
        """
        from sklearn.metrics.pairwise import cosine_similarity

        top_k       = top_k or C.RAG_TOP_K
        window_days = window_days or C.RAG_WINDOW_DAYS
        as_of       = pd.Timestamp(as_of).normalize()
        earliest    = as_of - timedelta(days=window_days)

        # THE look-ahead guard: date must be <= as_of.
        eligible = np.where((self.dates <= as_of) & (self.dates >= earliest))[0]
        if len(eligible) == 0:
            return []

        q_vec = self.vectorizer.transform([query])
        sims  = cosine_similarity(q_vec, self.matrix[eligible])[0]

        age_days = np.array([(as_of - self.dates[i]).days for i in eligible], dtype=float)
        recency  = 1.0 - 0.7 * (age_days / max(window_days, 1))     # 1.0 -> 0.3

        scores = sims * recency
        order  = np.argsort(-scores)[:top_k]

        results = []
        for pos in order:
            idx = eligible[pos]
            if scores[pos] <= 0:            # no word overlap at all - not evidence
                continue
            item = dict(self.items[idx])
            item["score"]      = round(float(scores[pos]), 4)
            item["similarity"] = round(float(sims[pos]), 4)
            item["age_days"]   = int(age_days[pos])
            results.append(item)
        return results


def market_state_query(bundle, ticker, date):
    """Build a retrieval query out of what is observable on `date`.

    Used by the backtest, where there is no human typing a question. It puts the
    current regime and recent move into words so TF-IDF has something to match.
    """
    from .data_loader import regime_of
    regime, vix_level = regime_of(bundle["vix"], date)

    feats = bundle["features"][ticker]
    prior = feats.index[feats.index <= pd.Timestamp(date)]
    if len(prior) == 0:
        return "market conditions volatility"

    row = feats.loc[prior[-1]]
    ret5 = float(row["ret_5d"]) * 100
    direction = "rising" if ret5 > 0 else "falling"

    return (
        f"market volatility VIX {vix_level:.0f} "
        f"{'stressed high volatility' if regime == 'SHOCK' else 'calm conditions'} "
        f"stock {direction} {abs(ret5):.1f} percent recent move "
        f"federal reserve rates inflation"
    )
