"""
MANALORE - Academy of Card Mastery (Streamlit build)
====================================================
A learning-first MTG card intelligence app for beginners and pros.

Tabs mirror the web app:
  Library  - the 1000 most-played cards, scored and filterable
  Lookup   - search any card with live suggestions
  Sets     - browse every release, with Play Booster vs Collector Booster views
  Builder  - pick cards from a set, we build the deck around them and name it
  Market   - demand-driven 90-day outlook across the library
  Academy  - how the scores are built, in plain language

Run:
  pip install -r requirements.txt
  streamlit run app.py

Data refreshes when the page loads. Prices are aggregated market values for
study, not a brokerage quote. Outlooks are analysis, not financial advice.
"""

import re
import math
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st

API = "https://api.scryfall.com"
HEADERS = {"User-Agent": "Manalore/1.0", "Accept": "application/json"}
FMTS = ["standard", "pioneer", "modern", "legacy", "vintage", "commander", "pauper"]
TARGET = 2500

# Hugging Face history dataset (built by collector.py via a daily GitHub Action).
# Set HF_DATASET_REPO in Streamlit secrets to enable real price history + forecasting.
# Reads are public for a public dataset; no token needed to read.
HF_REPO = None
try:
    HF_REPO = st.secrets.get("HF_DATASET_REPO")  # e.g. "you/manalore-mtg-history"
except Exception:
    HF_REPO = None


@st.cache_data(show_spinner=False, ttl=3600)
def load_price_history(repo, max_snapshots=120):
    """Load accumulated daily price snapshots from the Hugging Face dataset.
    Returns a tidy DataFrame [snapshot_date, name, set, usd] or None if unavailable.
    Reads only the slim columns needed, across available dated partitions."""
    if not repo:
        return None
    try:
        from huggingface_hub import HfApi
        import json as _json
        api = HfApi()
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
        parts = sorted([f for f in files if f.startswith("data/snapshot_date=") and f.endswith(".parquet")])
        parts = parts[-max_snapshots:]
        if not parts:
            return None
        frames = []
        for p in parts:
            url = f"https://huggingface.co/datasets/{repo}/resolve/main/{p}"
            try:
                # read only the columns we need to keep it light
                df = pd.read_parquet(url, columns=["snapshot_date", "name", "set", "prices"])
            except Exception:
                df = pd.read_parquet(url)
            # extract usd from the JSON-encoded prices field
            def _usd(x):
                try:
                    d = _json.loads(x) if isinstance(x, str) else (x or {})
                    v = d.get("usd") or d.get("usd_foil")
                    return float(v) if v else None
                except Exception:
                    return None
            df["usd"] = df["prices"].map(_usd)
            frames.append(df[["snapshot_date", "name", "set", "usd"]])
        hist = pd.concat(frames, ignore_index=True)
        hist = hist.dropna(subset=["usd"])
        return hist
    except Exception:
        return None


def card_history(hist, name):
    """Time-series of one card's price, sorted by date. Empty frame if none."""
    if hist is None:
        return pd.DataFrame(columns=["snapshot_date", "usd"])
    h = hist[hist["name"] == name][["snapshot_date", "usd"]].copy()
    if h.empty:
        return h
    h = h.groupby("snapshot_date", as_index=False)["usd"].mean().sort_values("snapshot_date")
    return h

# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def load_library(target=TARGET):
    """Fallback: the most-played cards from the live API (used when HF is not configured)."""
    out, url = [], (f"{API}/cards/search?q=" +
                    requests.utils.quote("-is:digital -t:basic legal:commander") +
                    "&order=edhrec&unique=cards")
    while url and len(out) < target:
        try:
            r = requests.get(url, headers=HEADERS, timeout=20)
            if not r.ok:
                break
            d = r.json()
        except requests.RequestException:
            break
        for c in d.get("data", []):
            if c.get("name") and not c.get("digital"):
                out.append(c)
        url = d.get("next_page") if d.get("has_more") else None
        if url:
            time.sleep(0.08)
    return out[:target]


@st.cache_data(show_spinner=False, ttl=1800)
def load_full_pool_from_hf(repo):
    """Load the FULL card library from the most recent HF snapshot.
    Returns a list of card dicts (no limit), or None if HF is not configured.
    Deduplicates by oracle_id so each unique card appears once.
    Falls back gracefully on any error."""
    if not repo:
        return None
    try:
        import json as _json
        from huggingface_hub import HfApi
        api = HfApi()
        files = sorted([f for f in api.list_repo_files(repo_id=repo, repo_type="dataset")
                        if f.startswith("data/snapshot_date=") and f.endswith(".parquet")])
        if not files:
            return None
        latest = files[-1]
        url = f"https://huggingface.co/datasets/{repo}/resolve/main/{latest}"
        needed = ["name", "oracle_id", "set", "set_name", "set_type", "released_at",
                  "oracle_text", "type_line", "cmc", "mana_cost", "colors",
                  "color_identity", "keywords", "legalities", "rarity", "reserved",
                  "prices", "edhrec_rank", "img_normal", "img_art_crop",
                  "foil", "nonfoil", "finishes", "frame_effects", "border_color",
                  "full_art", "promo_types", "layout", "collector_number", "digital"]
        try:
            df = pd.read_parquet(url, columns=[c for c in needed])
        except Exception:
            df = pd.read_parquet(url)
        # drop digital-only cards
        if "digital" in df.columns:
            df = df[df["digital"] != True]
        # parse JSON-encoded columns back to Python objects
        json_list_cols = ["colors", "color_identity", "keywords", "finishes",
                          "promo_types", "frame_effects"]
        json_dict_cols = ["prices", "legalities"]
        for col in json_list_cols:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: (_json.loads(x) if isinstance(x, str) else x) or [])
        for col in json_dict_cols:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda x: (_json.loads(x) if isinstance(x, str) else x) or {})
        # rebuild card dicts and deduplicate by oracle_id
        seen, cards = {}, []
        for row in df.to_dict("records"):
            if not row.get("name"):
                continue
            row["image_uris"] = {
                "normal": row.pop("img_normal", None),
                "art_crop": row.pop("img_art_crop", None),
            }
            key = row.get("oracle_id") or row.get("name")
            if key not in seen:
                seen[key] = True
                cards.append(row)
        return cards if cards else None
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=1800)
def scry_named(name):
    r = requests.get(f"{API}/cards/named", params={"fuzzy": name}, headers=HEADERS, timeout=20)
    return r.json() if r.ok else None


@st.cache_data(show_spinner=False, ttl=900)
def scry_autocomplete(q):
    r = requests.get(f"{API}/cards/autocomplete", params={"q": q}, headers=HEADERS, timeout=20)
    return (r.json().get("data", []) if r.ok else [])


@st.cache_data(show_spinner=False, ttl=900)
def scry_collection(names):
    out = []
    for i in range(0, len(names), 70):
        chunk = names[i:i + 70]
        try:
            r = requests.post(f"{API}/cards/collection",
                              json={"identifiers": [{"name": n} for n in chunk]},
                              headers=HEADERS, timeout=20)
            if r.ok:
                out.extend(r.json().get("data", []))
        except requests.RequestException:
            pass
        time.sleep(0.05)
    return out


@st.cache_data(show_spinner=False, ttl=900)
def scry_search(query, order="edhrec", unique="cards", limit=175):
    cards, url = [], f"{API}/cards/search"
    params = {"q": query, "order": order, "unique": unique}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if not r.ok:
            return []
        d = r.json()
        cards.extend(d.get("data", []))
        while d.get("has_more") and len(cards) < limit:
            time.sleep(0.08)
            r = requests.get(d["next_page"], headers=HEADERS, timeout=20)
            if not r.ok:
                break
            d = r.json()
            cards.extend(d.get("data", []))
    except requests.RequestException:
        return cards[:limit]
    return cards[:limit]


@st.cache_data(show_spinner=False, ttl=3600)
def load_sets():
    try:
        r = requests.get(f"{API}/sets", headers=HEADERS, timeout=20)
        if not r.ok:
            return []
        sets = [s for s in r.json().get("data", []) if not s.get("digital") and s.get("card_count", 0) > 0]
        sets.sort(key=lambda s: s.get("released_at", ""), reverse=True)
        return sets
    except requests.RequestException:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def load_full_set(code, unique="prints"):
    """Fetch every card in a set once, then we filter in memory for instant UX."""
    cards, url = [], f"{API}/cards/search"
    params = {"q": f"e:{code}", "order": "set", "unique": unique}
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=20)
        if not r.ok:
            return []
        d = r.json()
        cards.extend(d.get("data", []))
        pages = 0
        while d.get("has_more") and pages < 8:
            time.sleep(0.07)
            r = requests.get(d["next_page"], headers=HEADERS, timeout=20)
            if not r.ok:
                break
            d = r.json()
            cards.extend(d.get("data", []))
            pages += 1
    except requests.RequestException:
        return cards
    for c in cards:
        c["_f"] = features(c)
    return cards


@st.cache_data(show_spinner=False, ttl=3600)
def print_history(card_name):
    """All printings of a card across sets, newest first, with set + price."""
    try:
        r = requests.get(f"{API}/cards/search",
                         params={"q": f'!"{card_name}"', "unique": "prints", "order": "released"},
                         headers=HEADERS, timeout=20)
        if not r.ok:
            return []
        rows = []
        for c in r.json().get("data", []):
            rows.append({"Set": c.get("set_name", ""), "Code": c.get("set", "").upper(),
                         "Released": c.get("released_at", ""), "Rarity": c.get("rarity", "").title(),
                         "Price": price_now(c)})
        return rows
    except requests.RequestException:
        return []


# Strategy -> Scryfall oracle/type filters, used to search the WHOLE game
STRAT_FILTER = {
    "Aggro": '(t:creature mv<=3)',
    "Midrange": '(t:creature mv>=2 mv<=5 or o:"destroy target")',
    "Control": '(o:"counter target spell" or o:"draw" or o:"destroy target" or t:planeswalker)',
    "Combo": '(o:"whenever" or o:"you may" or t:artifact or o:"search your library")',
}
FORMAT_FILTER = {
    "Commander": "f:commander", "Modern": "f:modern", "Standard": "f:standard",
    "Pioneer": "f:pioneer", "Legacy": "f:legacy", "Pauper": "f:pauper", "Any": "",
}


@st.cache_data(show_spinner=False, ttl=1800)
def build_candidate_pool(colors_key, strategy, fmt, budget):
    """Search across ALL Magic cards for candidates that fit the deck brief.
    colors_key is a sorted string like 'GRU'. Pulls a deep, ranked pool."""
    parts = []
    if colors_key:
        parts.append(f"id<={colors_key.lower()}")
    else:
        parts.append("-t:land")
    parts.append(STRAT_FILTER.get(strategy, ""))
    parts.append(FORMAT_FILTER.get(fmt, ""))
    parts.append("-t:basic")
    if budget == "Budget":
        parts.append("usd<=5")
    elif budget == "Competitive":
        parts.append("(f:modern or f:legacy or f:commander)")
    q = " ".join(p for p in parts if p)
    return scry_search(q, order="edhrec", unique="cards", limit=260)


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def clamp(v, a, b):
    return max(a, min(b, v))


def img_uri(c, kind="normal"):
    u = c.get("image_uris") or (c.get("card_faces", [{}])[0].get("image_uris") if c.get("card_faces") else None)
    return u.get(kind) if u else None


def oracle(c):
    t = c.get("oracle_text", "")
    if not t and c.get("card_faces"):
        t = " ".join(f.get("oracle_text", "") for f in c["card_faces"])
    return (t or "").lower()


def type_line(c):
    return c.get("type_line") or (c.get("card_faces", [{}])[0].get("type_line", "") if c.get("card_faces") else "")


def _num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def price_now(c):
    p = c.get("prices", {}) or {}
    eur = _num(p.get("eur"))
    return _num(p.get("usd")) or _num(p.get("usd_foil")) or (eur * 1.08 if eur else None)


def price_info(c):
    """Return (value, source_label). Honest about whether it's normal, foil, or converted."""
    p = c.get("prices", {}) or {}
    if _num(p.get("usd")):
        return _num(p.get("usd")), "market"
    if _num(p.get("usd_foil")):
        return _num(p.get("usd_foil")), "foil only"
    if _num(p.get("eur")):
        return _num(p.get("eur")) * 1.08, "converted from EUR"
    return None, "unavailable"


def legal_formats(c):
    L = c.get("legalities", {})
    return [f for f in FMTS if L.get(f) == "legal"]


def fmt_usd(v):
    if v is None:
        return "n/a"
    if v < 100:
        return f"${v:,.2f}"
    return f"${round(v):,}"


# ---- card attribute helpers (for filtering / origin) ----
COLOR_NAMES = {"W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green"}


def has_foil(c):
    return "foil" in (c.get("finishes") or []) or bool(c.get("foil"))


def has_nonfoil(c):
    return "nonfoil" in (c.get("finishes") or []) or bool(c.get("nonfoil"))


def treatment(c):
    fe = c.get("frame_effects") or []
    pt = c.get("promo_types") or []
    if c.get("border_color") == "borderless":
        return "Borderless"
    if "showcase" in fe:
        return "Showcase"
    if "extendedart" in fe:
        return "Extended"
    if c.get("full_art"):
        return "Full art"
    if "serialized" in pt:
        return "Serialized"
    if "etched" in (c.get("finishes") or []):
        return "Etched"
    return "Standard"


def color_category(c):
    ci = c.get("color_identity", [])
    if not ci:
        return "Colorless"
    if len(ci) > 1:
        return "Multicolor"
    return COLOR_NAMES.get(ci[0], "Colorless")


def primary_type(c):
    tl = type_line(c).lower()
    for t in ["land", "creature", "instant", "sorcery", "artifact", "enchantment", "planeswalker", "battle"]:
        if t in tl:
            return t.capitalize()
    return "Other"


# ----------------------------------------------------------------------------
# MODEL (mirrors the web app exactly)
# ----------------------------------------------------------------------------
def features(c):
    t, tl = oracle(c), type_line(c).lower()
    cmc = c.get("cmc", 0) or 0
    rank = c.get("edhrec_rank") or 45000
    play_rank = clamp(10 - math.log10(rank + 1) * 2.05, 0, 10)
    ubiq = clamp(len(legal_formats(c)) * 1.55, 0, 10)
    kw = clamp(len(c.get("keywords", [])) * 1.6 + (2 if re.search(r"modal|choose one|escape|flashback", t) else 0), 0, 10)
    eff = clamp(9.5 - cmc * 1.1, 1, 10)
    if "instant" in tl:
        eff = clamp(eff + 0.8, 0, 10)
    card_adv = 0
    if re.search(r"draw (a|two|three|four|\w+) cards?", t): card_adv += 6
    if "search your library" in t: card_adv += 3
    if re.search(r"create .*token", t): card_adv += 2
    if re.search(r"whenever .* dies|whenever .* enters", t): card_adv += 2
    card_adv = clamp(card_adv, 0, 10)
    ci = c.get("color_identity", [])
    flex = clamp(4 + (2 - len(ci)) + (2 if "instant" in tl else 0) + (2 if re.search(r"any (color|type)|choose|modal", t) else 0), 0, 10)
    tempo = clamp((3 if re.search(r"haste|flash|flying|prowess", t) else 0) + (10 - cmc) * 0.55 + (1.5 if "creature" in tl else 0), 0, 10)
    return dict(play_rank=play_rank, ubiq=ubiq, kw=kw, eff=eff, card_adv=card_adv, flex=flex, tempo=tempo, cmc=cmc, rank=rank)


def reprint_risk(c):
    rr = {"common": 8, "uncommon": 7, "rare": 5, "mythic": 3, "special": 2, "bonus": 2}.get(c.get("rarity"), 5)
    return 0 if c.get("reserved") else rr


W_IMP = dict(play_rank=.26, eff=.18, card_adv=.14, flex=.12, ubiq=.18, kw=.12)


def feats(c):
    if "_f" not in c:
        c["_f"] = features(c)
    return c["_f"]


def enrich_card(c):
    feats(c)
    return c


def importance(c):
    f = feats(c)
    return round(sum(W_IMP[k] * f[k] for k in W_IMP) / 10 * 100)


def demand(c):
    f = feats(c)
    return round((0.55 * f["play_rank"] + 0.25 * f["flex"] + 0.20 * f["card_adv"]) / 10 * 100)


def drift_mo(c):
    raw = (demand(c) / 100 - 0.5) * 0.060 - (reprint_risk(c) / 10) * 0.020
    return clamp(raw, -0.04, 0.05)


def proj_price(c):
    p = price_now(c)
    if p is None:
        return None
    projected = p * (1 + 3 * drift_mo(c))
    return max(0.01, round(projected, 2))


def delta_pct(c):
    p, pj = price_now(c), proj_price(c)
    return 0 if (not p) else (pj - p) / p * 100


def confidence(c):
    f = feats(c)
    return round(clamp(45 + f["play_rank"] * 5 - reprint_risk(c) * 1.5, 45, 95))


def role(c):
    t, tl = oracle(c), type_line(c).lower()
    if "land" in tl: return "Land"
    if re.search(r"counter target (spell|ability|creature spell)", t): return "Counter"
    if re.search(r"destroy target|exile target|deals? \d+ damage to (any|target)|fight", t): return "Removal"
    if re.search(r"add \{|untap target land|search your library for .*(land|basic)", t): return "Ramp"
    if re.search(r"draw (a|two|three|four|\w+) cards?", t): return "Card Advantage"
    if re.search(r"discard|sacrifices|can't|loses the game|skip (your|their)", t): return "Disruption"
    if "planeswalker" in tl: return "Engine"
    if "artifact" in tl or "enchantment" in tl: return "Engine"
    if "creature" in tl: return "Threat"
    return "Spell"


# ============================================================================
# MACHINE LEARNING LAYER  (comprehensive upgrade)
# ============================================================================
# Three model families, all trained in-app on the scraped library + HF history:
#
# 1. SNAPSHOT MODELS (train on the 2,500-card live snapshot each load)
#    - Price model  : HistGradientBoosting -> fair value from card attributes
#    - Importance   : GradientBoosting     -> learned power score
#    - Similarity   : NearestNeighbors cosine -> synergy / "plays like this"
#
# 2. TIME-SERIES FORECASTER (trains on accumulated HF price history)
#    - LightGBM / GradientBoosting -> predicts 30-day price change
#    - Features: momentum (7d/14d/30d), volatility, price level, card attributes
#    - Validated with time-based splits, not random splits
#    - Replaces the heuristic drift formula when enough history exists
#
# 3. MARKET MOMENTUM SIGNALS (derived from history per card)
#    - 7d/14d/30d momentum, volatility, trend, value opportunity score
#    - Used in market tracker and card popup
# ============================================================================

ML_TYPES = ["creature", "instant", "sorcery", "artifact", "enchantment", "planeswalker", "land", "battle"]
ML_TEXT_SIGNALS = [
    ("draws",     r"draw (a|two|three|four|\w+) cards?"),
    ("removal",   r"destroy target|exile target|deals? \d+ damage"),
    ("counter",   r"counter target"),
    ("ramp",      r"add \{|search your library for .*(land|basic)"),
    ("token",     r"create .*token"),
    ("recursion", r"return .* from your graveyard"),
    ("tutor",     r"search your library"),
    ("etb",       r"when .* enters"),
    ("haste",     r"\bhaste\b"),
    ("draw_step", r"at the beginning of .* draw"),
]

# ---- card attribute features (used by snapshot + forecaster models) ----
def ml_feature_row(c):
    tl = type_line(c).lower()
    t = oracle(c)
    cmc = float(c.get("cmc", 0) or 0)
    ci = c.get("color_identity", [])
    rank = c.get("edhrec_rank") or 60000
    rar = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3, "special": 3, "bonus": 3}.get(c.get("rarity"), 1)
    fmts = len(legal_formats(c))
    row = {
        "cmc": cmc,
        "cmc_sq": cmc ** 2,
        "log_rank": math.log10(rank + 1),
        "n_colors": len(ci),
        "n_formats": fmts,
        "n_formats_sq": fmts ** 2,
        "n_keywords": len(c.get("keywords", [])),
        "rarity_ord": rar,
        "reserved": 1 if c.get("reserved") else 0,
        "text_len": min(400, len(t)) / 400.0,
        "is_multicolor": 1 if len(ci) > 1 else 0,
        "is_colorless": 1 if not ci else 0,
    }
    for name in ML_TYPES:
        row[f"is_{name}"] = 1 if name in tl else 0
    for name, pat in ML_TEXT_SIGNALS:
        row[f"sig_{name}"] = 1 if re.search(pat, t) else 0
    return row


ML_FEATURE_ORDER = (
    ["cmc", "cmc_sq", "log_rank", "n_colors", "n_formats", "n_formats_sq",
     "n_keywords", "rarity_ord", "reserved", "text_len", "is_multicolor", "is_colorless"]
    + [f"is_{n}" for n in ML_TYPES]
    + [f"sig_{n}" for n, _ in ML_TEXT_SIGNALS]
)

ML_FEATURE_LABELS = {
    "cmc": "Mana cost", "cmc_sq": "Mana cost (sq)", "log_rank": "Play-rate",
    "n_colors": "Colors", "n_formats": "Format breadth", "n_formats_sq": "Format breadth (sq)",
    "n_keywords": "Keywords", "rarity_ord": "Rarity", "reserved": "Reserved list",
    "text_len": "Text length", "is_multicolor": "Multicolor", "is_colorless": "Colorless",
    **{f"is_{n}": f"Type: {n}" for n in ML_TYPES},
    **{f"sig_{n}": f"Signal: {n}" for n, _ in ML_TEXT_SIGNALS},
}


# ---- time-series features per card from price history ----
def ts_features(hist, name):
    """Rolling momentum, volatility, trend from a card's price history.
    Returns a dict of floats, or zeros if not enough history."""
    zero = {"mom_7d": 0.0, "mom_14d": 0.0, "mom_30d": 0.0,
            "vol_14d": 0.0, "vol_30d": 0.0, "log_price": 0.0,
            "price_range_ratio": 0.0, "n_obs": 0.0, "has_ts": 0.0}
    h = card_history(hist, name) if hist is not None else None
    if h is None or len(h) < 5:
        return zero
    p = h["usd"].values.astype(float)
    n = len(p)
    last = p[-1]
    def mom(k):
        return float(last / p[max(0, n - k)] - 1) if n >= k else 0.0
    def vol(k):
        chunk = p[max(0, n - k):]
        return float(np.std(np.diff(np.log1p(chunk)))) if len(chunk) > 1 else 0.0
    return {
        "mom_7d": mom(7), "mom_14d": mom(14), "mom_30d": mom(30),
        "vol_14d": vol(14), "vol_30d": vol(30),
        "log_price": float(np.log1p(last)),
        "price_range_ratio": float(p[max(0, n-7):].max() / (p[max(0, n-7):].min() + 1e-6) - 1),
        "n_obs": float(min(n, 120)), "has_ts": 1.0,
    }

TS_FEATURE_ORDER = ["mom_7d", "mom_14d", "mom_30d", "vol_14d", "vol_30d",
                    "log_price", "price_range_ratio", "n_obs", "has_ts"]


# ---- snapshot models (trained once per library load) ----
@st.cache_resource(show_spinner=False)
def train_models(_pool_sig, rows, prices, ranks):
    out = {"ok": False}
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
    except Exception:
        return out

    X = np.array([[r[k] for k in ML_FEATURE_ORDER] for r in rows], dtype=float)
    n_total = len(X)

    # ---- price model ----
    price_idx = [i for i, p in enumerate(prices) if p and p > 0]
    if len(price_idx) >= 150:
        Xp = X[price_idx]
        yp = np.log1p(np.array([prices[i] for i in price_idx], dtype=float))
        price_model = HistGradientBoostingRegressor(
            max_depth=5, max_iter=300, learning_rate=0.05,
            l2_regularization=0.8, min_samples_leaf=15, random_state=7)
        price_model.fit(Xp, yp)
        # CV on a sample so large pools don't time out
        cv_n = min(len(Xp), 8000)
        idx_cv = np.random.RandomState(42).choice(len(Xp), cv_n, replace=False)
        try:
            r2 = float(np.mean(cross_val_score(price_model, Xp[idx_cv], yp[idx_cv],
                                                cv=5, scoring="r2")))
        except Exception:
            r2 = float("nan")
        out["price_model"] = price_model
        out["price_r2"] = r2
        out["price_n"] = len(price_idx)
        try:
            out["price_feat_imp"] = dict(zip(ML_FEATURE_ORDER,
                                             price_model.feature_importances_.tolist()))
        except Exception:
            pass

    # ---- importance model (HistGBM scales to 100K+) ----
    inv_rank = 10 - np.clip(np.log10(np.array(ranks, dtype=float) + 1) * 2.05, 0, 10)
    breadth = np.clip(np.array([r["n_formats"] for r in rows]) * 1.55, 0, 10)
    y_imp = np.clip((0.7 * inv_rank + 0.3 * breadth) / 10 * 100, 0, 100)
    imp_model = HistGradientBoostingRegressor(
        max_depth=4, max_iter=260, learning_rate=0.05,
        l2_regularization=0.5, min_samples_leaf=12, random_state=7)
    imp_model.fit(X, y_imp)
    cv_n2 = min(n_total, 8000)
    idx_cv2 = np.random.RandomState(42).choice(n_total, cv_n2, replace=False)
    try:
        imp_r2 = float(np.mean(cross_val_score(imp_model, X[idx_cv2], y_imp[idx_cv2],
                                                cv=5, scoring="r2")))
    except Exception:
        imp_r2 = float("nan")
    out["imp_model"] = imp_model
    out["imp_r2"] = imp_r2
    try:
        out["imp_feat_imp"] = dict(zip(ML_FEATURE_ORDER,
                                       imp_model.feature_importances_.tolist()))
    except Exception:
        pass

    # ---- similarity (NearestNeighbors handles 100K fine) ----
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    nn = NearestNeighbors(n_neighbors=min(15, n_total), metric="cosine",
                          algorithm="brute").fit(Xs)
    out["scaler"] = scaler
    out["nn"] = nn
    out["X"] = X
    out["ok"] = True
    return out


@st.cache_data(show_spinner=False)
def build_ml(_pool_sig, names, rows, prices, ranks):
    models = train_models(_pool_sig, rows, prices, ranks)
    return models, names


# ---- time-series forecaster (trained on HF history) ----
@st.cache_resource(show_spinner=False)
def train_forecaster(_hist_sig, hist_parquet_rows, attr_rows_by_name):
    """Train a 30-day price forecaster on accumulated price history.
    Creates sliding-window training examples: features at day t -> return at t+30.
    Uses time-based validation (older data trains, newest 20% tests) for honest metrics."""
    out = {"ok": False}
    if not hist_parquet_rows or len(hist_parquet_rows) < 80:
        return out
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import r2_score, mean_absolute_error
    except Exception:
        return out

    # rebuild a name -> sorted price list map from the compact row list
    name_prices = {}
    for row in hist_parquet_rows:
        name_prices.setdefault(row["name"], []).append((row["date"], float(row["usd"])))
    for nm in name_prices:
        name_prices[nm].sort(key=lambda x: x[0])

    Xts, yts = [], []
    HORIZON = 30
    for nm, series in name_prices.items():
        if len(series) < HORIZON + 7:
            continue
        attrs = attr_rows_by_name.get(nm)
        if attrs is None:
            continue
        prices_arr = np.array([v for _, v in series], dtype=float)
        n = len(prices_arr)
        for i in range(7, n - HORIZON):
            p_now = prices_arr[i]
            p_future = prices_arr[i + HORIZON]
            if p_now <= 0 or p_future <= 0:
                continue
            target = float(np.log(p_future / p_now))
            mom7 = float(p_now / prices_arr[max(0, i-7)] - 1)
            mom14 = float(p_now / prices_arr[max(0, i-14)] - 1) if i >= 14 else 0.0
            mom30 = float(p_now / prices_arr[max(0, i-30)] - 1) if i >= 30 else 0.0
            chunk = prices_arr[max(0, i-14):i+1]
            vol14 = float(np.std(np.diff(np.log1p(chunk)))) if len(chunk) > 1 else 0.0
            feat = [math.log10(i + 1), float(np.log1p(p_now)),
                    mom7, mom14, mom30, vol14] + [attrs.get(k, 0.0) for k in ML_FEATURE_ORDER]
            Xts.append(feat)
            yts.append(target)

    if len(Xts) < 50:
        return out

    Xarr = np.array(Xts, dtype=float)
    yarr = np.array(yts, dtype=float)
    # time-based split: train on oldest 80%, test on newest 20%
    split = int(len(Xarr) * 0.8)
    Xtr, Xte = Xarr[:split], Xarr[split:]
    ytr, yte = yarr[:split], yarr[split:]

    model = GradientBoostingRegressor(
        max_depth=3, n_estimators=220, learning_rate=0.05,
        subsample=0.8, random_state=7)
    model.fit(Xtr, ytr)

    out["forecaster"] = model
    out["n_examples"] = len(Xarr)
    out["n_cards"] = len(name_prices)
    if len(Xte) >= 10:
        ypred = model.predict(Xte)
        out["test_r2"] = round(float(r2_score(yte, ypred)), 3)
        out["test_mae"] = round(float(mean_absolute_error(yte, ypred)), 4)
        # directional accuracy: did we predict up/down correctly?
        dir_acc = float(np.mean(np.sign(ypred) == np.sign(yte)))
        out["dir_acc"] = round(dir_acc, 3)
    out["ok"] = True
    return out


def forecast_30d(forecaster_models, c, hist):
    """Predict 30-day log-return for a card. Returns (predicted_pct, confidence_label) or None."""
    if not forecaster_models.get("ok") or "forecaster" not in forecaster_models:
        return None
    tsf = ts_features(hist, c["name"])
    if tsf["has_ts"] == 0:
        return None
    attrs = ml_feature_row(c)
    p = price_now(c) or 0
    feat = [math.log10(max(1, tsf["n_obs"])), float(np.log1p(p)),
            tsf["mom_7d"], tsf["mom_14d"], tsf["mom_30d"], tsf["vol_14d"]
            ] + [attrs.get(k, 0.0) for k in ML_FEATURE_ORDER]
    Xf = np.array([feat], dtype=float)
    log_ret = float(forecaster_models["forecaster"].predict(Xf)[0])
    pct = (math.exp(log_ret) - 1) * 100
    conf_lbl = ("High" if abs(pct) > 8 else "Medium" if abs(pct) > 3 else "Low")
    return round(pct, 1), conf_lbl


def momentum_signal(hist, name):
    """Compute momentum + volatility summary for a card. Returns dict."""
    tsf = ts_features(hist, name)
    trend = ("rising" if tsf["mom_7d"] > 0.04 else
             "falling" if tsf["mom_7d"] < -0.04 else "flat")
    vol_lbl = ("volatile" if tsf["vol_14d"] > 0.08 else
               "stable" if tsf["vol_14d"] < 0.02 else "normal")
    return {"trend": trend, "vol_lbl": vol_lbl, **tsf}


# ---- snapshot ML inference functions ----
def ml_fair_value(models, c):
    if not models.get("ok") or "price_model" not in models:
        return None
    X = np.array([[ml_feature_row(c)[k] for k in ML_FEATURE_ORDER]], dtype=float)
    return max(0.01, round(float(np.expm1(models["price_model"].predict(X)[0])), 2))


def ml_importance(models, c):
    if not models.get("ok") or "imp_model" not in models:
        return importance(c)
    X = np.array([[ml_feature_row(c)[k] for k in ML_FEATURE_ORDER]], dtype=float)
    return int(clamp(round(float(models["imp_model"].predict(X)[0])), 0, 100))


def ml_value_signal(models, c):
    fair = ml_fair_value(models, c)
    p = price_now(c)
    if fair is None or not p:
        return None
    pct = (p - fair) / fair * 100
    verdict = ("overpriced" if pct > 35 else "underpriced" if pct < -35 else "fairly priced")
    return fair, verdict, pct


def ml_similar(models, names, idx, k=8):
    if not models.get("ok") or "nn" not in models:
        return []
    Xs = models["scaler"].transform(models["X"][idx:idx + 1])
    dist, ind = models["nn"].kneighbors(Xs, n_neighbors=min(k + 1, len(names)))
    return [int(j) for j in ind[0] if int(j) != idx][:k]


def ml_vector(models, c):
    if not models.get("ok") or "scaler" not in models:
        return None
    x = np.array([[ml_feature_row(c)[k] for k in ML_FEATURE_ORDER]], dtype=float)
    return models["scaler"].transform(x)[0]


def ml_anchor_centroid(models, anchors):
    vecs = [ml_vector(models, a) for a in anchors]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def ml_synergy_to(models, centroid, c):
    if centroid is None:
        return 0.5
    v = ml_vector(models, c)
    if v is None:
        return 0.5
    denom = np.linalg.norm(centroid) * np.linalg.norm(v)
    if denom == 0:
        return 0.5
    return (float(np.dot(centroid, v) / denom) + 1) / 2


def tier_word(im):
    return "a cornerstone" if im >= 80 else "a staple" if im >= 65 else "a solid role-player" if im >= 50 else "a niche pick"


def plain_read(c):
    im, d = importance(c), delta_pct(c)
    body = ("A cornerstone of competitive play. You will see this card everywhere, and demand keeps it valuable." if im >= 80 else
            "A widely played staple, a safe and useful card to own." if im >= 65 else
            "A solid role-player that shines in the right deck, though it is not an every-deck card." if im >= 50 else
            "A niche or casual pick. Its value is situational and depends on the deck.")
    direction = ("Our model leans bullish on its price over the next 90 days." if d >= 2 else
                 "Our model leans cautious on its price over the next 90 days." if d <= -2 else
                 "Our model sees its price holding roughly steady.")
    return f"This is {tier_word(im)}. {body} {direction}"


def analyst_notes(c):
    f, out, fmts = feats(c), [], legal_formats(c)
    if len(fmts) >= 4:
        out.append(("strong", f"Legal in {len(fmts)} formats. Demand rests on a broad base, not one metagame."))
    elif len(fmts) >= 2:
        out.append(("strong", f"Played across {', '.join(fmts)}, a reliable demand floor."))
    elif len(fmts) == 1:
        out.append(("caution", "Legal in one format only, so value leans on that format staying relevant."))
    else:
        out.append(("note", "Limited tournament legality; value is driven by casual and collector demand."))
    if c.get("edhrec_rank") and c["edhrec_rank"] < 800:
        out.append(("strong", f"Ranked #{c['edhrec_rank']:,} among the most-played cards in the game."))
    elif c.get("edhrec_rank"):
        out.append(("note", f"Solid play volume (rank #{c['edhrec_rank']:,}), but not top-tier ubiquity."))
    if f["eff"] >= 8.5:
        out.append(("strong", f"Elite rate for its cost (mana value {int(f['cmc'])}), the efficiency that decides what is playable."))
    if f["card_adv"] >= 6:
        out.append(("strong", "Generates card advantage on its own, exactly what grindy decks pay up for."))
    if f["flex"] >= 8:
        out.append(("strong", "Very flexible: low color commitment and broad use, so it slots into many decks."))
    if c.get("reserved"):
        out.append(("strong", "Can never be reprinted, so supply is locked. A structural floor under the price."))
    elif reprint_risk(c) >= 7:
        out.append(("caution", f"High supply risk ({c.get('rarity')}). New printings can expand supply and push the price down."))
    p = price_now(c)
    if p and p >= 150:
        out.append(("note", f"Premium price ({fmt_usd(p)}) shrinks the buyer pool, so moves are slower but stickier."))
    out.append(("note", f"Functional role read from the rules text: {role(c)}."))
    return out


# ----------------------------------------------------------------------------
# DECK NAMING
# ----------------------------------------------------------------------------
GUILDS = {"W": "Mono-White", "U": "Mono-Blue", "B": "Mono-Black", "R": "Mono-Red", "G": "Mono-Green",
          "WU": "Azorius", "WB": "Orzhov", "WR": "Boros", "WG": "Selesnya", "UB": "Dimir", "UR": "Izzet",
          "UG": "Simic", "BR": "Rakdos", "BG": "Golgari", "RG": "Gruul", "WUB": "Esper", "WUR": "Jeskai",
          "WUG": "Bant", "WBR": "Mardu", "WBG": "Abzan", "WRG": "Naya", "UBR": "Grixis", "UBG": "Sultai",
          "URG": "Temur", "BRG": "Jund", "WUBR": "Yore-Tiller", "WUBG": "Witch-Maw", "WURG": "Ink-Treader",
          "WBRG": "Dune-Brood", "UBRG": "Glint-Eye", "WUBRG": "Rainbow"}
DPREF = ["Emberfang", "Skyward", "Tidecaller", "Gravebound", "Sunlit", "Stormwrought", "Verdant", "Ashen",
         "Gilded", "Umbral", "Radiant", "Thornclad", "Frostbound", "Bloodsworn", "Eternal", "Wildfire",
         "Hallowed", "Duskborn", "Ironscale", "Mistveil"]
DNOUN = ["Doctrine", "Onslaught", "Covenant", "Gambit", "Vanguard", "Ascendancy", "Crusade", "Resonance",
         "Procession", "Uprising", "Sovereignty", "Reckoning", "Symphony", "Conclave", "Paradigm"]
ARCH_OF = {"Threat": "Aggro", "Removal": "Control", "Counter": "Control", "Card Advantage": "Midrange",
           "Engine": "Midrange", "Ramp": "Ramp", "Disruption": "Midrange", "Land": "Midrange", "Spell": "Midrange"}


def guild_name(cols):
    key = "".join(c for c in ["W", "U", "B", "R", "G"] if c in cols)
    return GUILDS.get(key, key or "Colorless")


def deck_name(cols, arch):
    g = guild_name(cols)
    seed = sum(ord(ch) for ch in (arch + "".join(cols)))
    return f"{g} {DPREF[seed % len(DPREF)]} {DNOUN[(seed * 7) % len(DNOUN)]}"


# ----------------------------------------------------------------------------
# DECK INTELLIGENCE ENGINE
# ----------------------------------------------------------------------------
# Recommended non-land role mix per strategy (as fractions of the spell slots),
# and target land counts. Used to build synergy-first decks and to grade them.
STRAT_PROFILE = {
    "Aggro":    {"roles": {"Threat": .62, "Removal": .20, "Card Advantage": .10, "Disruption": .08},
                 "curve": {0: .05, 1: .26, 2: .34, 3: .22, 4: .09, 5: .04}, "lands60": 22, "lands100": 34, "tagline": "go fast and close before the opponent stabilizes"},
    "Midrange": {"roles": {"Threat": .42, "Removal": .26, "Card Advantage": .16, "Ramp": .08, "Disruption": .08},
                 "curve": {1: .10, 2: .26, 3: .26, 4: .20, 5: .12, 6: .06}, "lands60": 24, "lands100": 37, "tagline": "trade efficiently, then win with stronger cards"},
    "Control":  {"roles": {"Removal": .30, "Counter": .22, "Card Advantage": .30, "Engine": .12, "Threat": .06},
                 "curve": {1: .12, 2: .26, 3: .24, 4: .18, 5: .12, 6: .08}, "lands60": 26, "lands100": 38, "tagline": "answer everything, win late with inevitability"},
    "Combo":    {"roles": {"Engine": .30, "Card Advantage": .26, "Ramp": .20, "Disruption": .14, "Counter": .10},
                 "curve": {1: .16, 2: .28, 3: .26, 4: .16, 5: .08, 6: .06}, "lands100": 36, "lands60": 23, "tagline": "assemble the engine, protect it, and combo off"},
}


def deck_lands(strategy, deck_size):
    prof = STRAT_PROFILE.get(strategy, STRAT_PROFILE["Midrange"])
    return prof["lands100"] if deck_size >= 100 else prof["lands60"]


def role_targets(strategy, spell_slots):
    """Recommended card counts by role for the chosen strategy."""
    prof = STRAT_PROFILE.get(strategy, STRAT_PROFILE["Midrange"])
    return {r: max(1, round(frac * spell_slots)) for r, frac in prof["roles"].items()}


def synergy_score(cards, strategy, models=None):
    """0-100. Rewards cards whose role fits the strategy and shared colors, and (when the ML
    model is available) how tightly the deck clusters in learned feature space."""
    if not cards:
        return 0
    prof = STRAT_PROFILE.get(strategy, STRAT_PROFILE["Midrange"])
    wanted = set(prof["roles"].keys())
    on_plan = sum(1 for c in cards if (("Other" if role(c) == "Spell" else role(c)) in wanted or "land" in type_line(c).lower()))
    on_plan_frac = on_plan / len(cards)
    cols = set()
    for c in cards:
        cols.update(c.get("color_identity", []))
    cohesion = 1.0 if len(cols) <= 2 else 0.85 if len(cols) == 3 else 0.65 if len(cols) == 4 else 0.5
    # ML cohesion: average similarity of each nonland card to the deck centroid
    ml_cohesion = None
    if models and models.get("ok") and "scaler" in models:
        nonland = [c for c in cards if "land" not in type_line(c).lower()]
        cen = ml_anchor_centroid(models, nonland)
        if cen is not None and nonland:
            sims = [ml_synergy_to(models, cen, c) for c in nonland]
            ml_cohesion = float(np.mean(sims))
    if ml_cohesion is not None:
        # blend: role fit, color cohesion, and learned feature cohesion
        return round(clamp(on_plan_frac * 50 + cohesion * 18 + ml_cohesion * 32, 0, 100))
    return round(clamp(on_plan_frac * 78 + cohesion * 22, 0, 100))


def consistency_score(cards, strategy, deck_size):
    """0-100. Rewards a smooth curve vs the target, and proper quantities over singletons."""
    spells = [c for c in cards if "land" not in type_line(c).lower()]
    if not spells:
        return 0
    prof = STRAT_PROFILE.get(strategy, STRAT_PROFILE["Midrange"])
    target = prof["curve"]
    actual = {}
    for c in spells:
        b = min(6, int(c.get("cmc", 0)))
        actual[b] = actual.get(b, 0) + 1
    n = len(spells)
    # curve distance (lower is better)
    dist = 0
    for b in range(7):
        want = target.get(b, 0)
        have = actual.get(b, 0) / n
        dist += abs(want - have)
    curve_fit = clamp(1 - dist, 0, 1)
    # quantity: singleton-heavy 60-card decks are less consistent; commander is singleton by rule
    if deck_size < 100:
        names = [c["name"] for c in spells]
        from collections import Counter
        cnt = Counter(names)
        multi = sum(v for v in cnt.values() if v >= 2)
        qty = clamp(multi / max(1, len(spells)), 0, 1)
        return round(clamp(curve_fit * 65 + qty * 35, 0, 100))
    return round(clamp(curve_fit * 100, 0, 100))


def curve_score(cards, strategy):
    return consistency_score(cards, strategy, 60)  # curve component reused


def budget_label(value):
    return "Budget" if value <= 50 else "Mid" if value <= 200 else "High-end"


def win_condition(cards, strategy):
    """Pick the most plausible win condition from the deck's contents."""
    def uniq_names(lst, k):
        seen, out = set(), []
        for c in lst:
            if c["name"] not in seen:
                seen.add(c["name"]); out.append(c["name"])
            if len(out) >= k:
                break
        return out
    threats = sorted([c for c in cards if role(c) == "Threat"], key=lambda c: -importance(c))
    engines = [c for c in cards if role(c) == "Engine"]
    if strategy == "Aggro":
        top = ", ".join(uniq_names(threats, 3))
        return f"Win by attacking fast with cheap threats like {top or 'your creatures'} before the opponent stabilizes."
    if strategy == "Control":
        return ("Win late: answer every threat, pull ahead on cards, then close with a few resilient finishers like "
                + (", ".join(uniq_names(threats, 2)) or "your top-end threats") + ".")
    if strategy == "Combo":
        return ("Win by assembling your engine (" + (", ".join(uniq_names(engines, 2)) or "key pieces")
                + "), protecting it, and converting it into a game-ending loop or burst.")
    top = ", ".join(uniq_names(threats, 3))
    return f"Win by trading efficiently, then taking over with stronger midrange threats like {top or 'your best creatures'}."


def deck_quality(cards, strategy, deck_size, value, brief_match, models=None):
    syn = synergy_score(cards, strategy, models)
    con = consistency_score(cards, strategy, deck_size)
    avg_imp = np.mean([importance(c) for c in cards]) if cards else 0
    # budget efficiency: power per dollar, normalized
    bud_eff = clamp((avg_imp / max(8, (value / max(1, len(cards))) + 8)) * 22, 0, 100)
    overall = round(0.30 * syn + 0.24 * con + 0.20 * avg_imp + 0.12 * bud_eff + 0.14 * brief_match)
    return {"synergy": syn, "consistency": con, "power": round(avg_imp),
            "budget_eff": round(bud_eff), "alignment": round(brief_match), "overall": clamp(overall, 0, 100)}


def decklist_csv(used, lands, cols):
    """Return a CSV string: Qty,Card,Role,Type,MV,Price."""
    from collections import Counter
    lines = ["Qty,Card,Role,Type,ManaValue,PriceUSD"]
    cnt = Counter(c["name"] for c in used)
    seen = set()
    for c in used:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        nm = c["name"].replace('"', "'")
        lines.append(f'{cnt[c["name"]]},"{nm}",{role(c)},"{type_line(c)}",{int(c.get("cmc",0))},{price_now(c) or 0:.2f}')
    lines.append(f'{lands},"Lands ({"".join(cols) or "C"})",Land,Land,0,0.00')
    return "\n".join(lines)


def deck_report_txt(name, brief, q, used, lands, value, cols, strategy, deck_size):
    from collections import Counter
    role_counts = Counter(("Other" if role(c) == "Spell" else role(c)) for c in used)
    lines = [
        f"MANALORE DECK REPORT", "=" * 48, name, brief, "",
        f"Format size: {len(used) + lands} cards ({len(used)} spells / {lands} lands)",
        f"Colors: {guild_name(cols)}    Est. value: ${value:.0f} ({budget_label(value)})", "",
        "QUALITY SCORES (0-100)",
        f"  Overall        {q['overall']}",
        f"  Synergy        {q['synergy']}",
        f"  Consistency    {q['consistency']}",
        f"  Avg power      {q['power']}",
        f"  Budget eff.    {q['budget_eff']}",
        f"  Brief match    {q['alignment']}", "",
        "WIN CONDITION", "  " + win_condition(used, strategy), "",
        "ROLE MIX",
    ]
    for r, n in role_counts.most_common():
        lines.append(f"  {r}: {n}")
    lines += ["", "DECKLIST"]
    seen = set()
    cnt = Counter(c["name"] for c in used)
    for c in used:
        if c["name"] in seen:
            continue
        seen.add(c["name"])
        lines.append(f"  {cnt[c['name']]}x {c['name']}  (PWR {importance(c)}, ${price_now(c) or 0:.2f})")
    lines.append(f"  {lands}x Lands")
    lines += ["", "Prices are aggregated market values for study, not a quote. Analysis is not financial advice."]
    return "\n".join(lines)


def assemble_deck(starters, candidates, strategy, deck_size, models=None):
    """ML-driven synergy build: fill role quotas, but rank candidates within each role by a
    blend of learned power and cosine similarity to the deck's evolving fingerprint, so the
    deck coheres around the build-around cards instead of being unrelated good cards.
    For 60-card decks, run impactful nonland spells as playsets for consistency."""
    lands = deck_lands(strategy, deck_size)
    spell_slots = deck_size - lands
    targets = role_targets(strategy, spell_slots)

    use_ml = bool(models and models.get("ok") and "scaler" in models)

    # bucket candidates by role
    buckets = {}
    for c in candidates:
        if "land" in type_line(c).lower():
            continue
        r = "Other" if role(c) == "Spell" else role(c)
        buckets.setdefault(r, []).append(c)

    used, used_names = [], set()

    # the deck "fingerprint" starts from the seed cards and updates as we add
    anchors = [s for s in starters]
    centroid = ml_anchor_centroid(models, anchors) if use_ml else None

    def rank(pool):
        """Order a role's candidates by 65% power + 35% synergy to current centroid."""
        if not use_ml or centroid is None:
            return sorted(pool, key=lambda c: -importance(c))
        def score(c):
            powr = importance(c) / 100.0
            syn = ml_synergy_to(models, centroid, c)
            return -(0.62 * powr + 0.38 * syn)
        return sorted(pool, key=score)

    def add(card, copies):
        nonlocal centroid
        if card["name"] in used_names or "land" in type_line(card).lower():
            return 0
        copies = max(1, min(copies, (4 if deck_size < 100 else 1)))
        added = 0
        for _ in range(copies):
            if len(used) >= spell_slots:
                break
            used.append(card); added += 1
        used_names.add(card["name"])
        # update the fingerprint with each distinct card so the deck stays cohesive
        if use_ml and added:
            anchors.append(card)
            centroid = ml_anchor_centroid(models, anchors)
        return added

    # 1) seed the user's build-around / favourite cards first
    for c in starters:
        add(c, 4 if deck_size < 100 else 1)

    # 2) fill each role toward its target, re-ranking by synergy as the deck grows
    per_copy = 3 if deck_size < 100 else 1
    for r, want in sorted(targets.items(), key=lambda kv: -kv[1]):
        have = sum(1 for c in used if ("Other" if role(c) == "Spell" else role(c)) == r)
        guard = 0
        while have < want and len(used) < spell_slots and guard < 200:
            guard += 1
            pool = [c for c in buckets.get(r, []) if c["name"] not in used_names]
            if not pool:
                break
            best = rank(pool)[0]
            got = add(best, per_copy)
            have += got

    # 3) top up remaining slots with the best on-plan, on-synergy cards left
    while len(used) < spell_slots:
        pool = [c for r in buckets for c in buckets[r] if c["name"] not in used_names]
        if not pool:
            break
        add(rank(pool)[0], 1)

    return used, lands, spell_slots, targets


def validate_deck(used, lands, targets, strategy, deck_size, cols, color_pick):
    """Return a list of (ok, message) checks confirming the deck matches the brief."""
    checks = []
    total = len(used) + lands
    checks.append((total == deck_size, f"Deck size is {total} (target {deck_size})."))
    spells = len(used)
    checks.append((spells >= deck_size - lands - 2, f"{spells} spells filled toward {deck_size - lands}."))
    # role coverage
    from collections import Counter
    rc = Counter(("Other" if role(c) == "Spell" else role(c)) for c in used)
    covered = sum(1 for r in targets if rc.get(r, 0) >= max(1, int(targets[r] * 0.5)))
    checks.append((covered >= max(1, len(targets) - 1),
                   f"{covered}/{len(targets)} core roles for {strategy} are adequately filled."))
    # color match
    if color_pick:
        deck_cols = set()
        for c in used:
            deck_cols.update(c.get("color_identity", []))
        ok = deck_cols.issubset(set(color_pick))
        checks.append((ok, "All cards fit your chosen colors." if ok else "Some cards fall outside your chosen colors."))
    return checks


# ----------------------------------------------------------------------------
# THEME
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MANALORE", page_icon="✦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Spectral:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.stApp{background:linear-gradient(180deg,#0b0e12,#0e1319);color:#ece6d6;font-family:'Spectral',serif}
/* collapse Streamlit chrome for more usable space */
#MainMenu{visibility:hidden}
header[data-testid="stHeader"]{height:0;background:transparent}
[data-testid="stToolbar"]{display:none!important}
[data-testid="stToolbarActions"]{display:none!important}
[data-testid="stActionButtonIcon"]{display:none!important}
[data-testid="stDecoration"]{display:none!important}
[data-testid="stAppDeployButton"]{display:none!important}
.stAppDeployButton{display:none!important}
.stActionButton{display:none!important}
footer{visibility:hidden}
.block-container{padding-top:1.2rem;padding-bottom:2rem;max-width:1400px}
h1,h2,h3{font-family:'Cinzel',serif!important;color:#f0d292!important;letter-spacing:1.5px}
[data-testid="stMetricValue"]{font-family:'IBM Plex Mono',monospace;color:#f0d292}
.stTabs [data-baseweb="tab"]{font-family:'IBM Plex Mono';letter-spacing:1px;text-transform:uppercase;font-size:12px}
.cap{font-family:'IBM Plex Mono';font-size:11px;color:#8c97a5}
.gold{color:#f0d292}.up{color:#5fc28a}.dn{color:#df7261}
.verdict{background:linear-gradient(135deg,rgba(217,168,80,.12),rgba(120,92,168,.08));border:1px solid #33404e;border-left:3px solid #d9a850;border-radius:12px;padding:14px 16px;font-size:15px;line-height:1.6}
.hero{background:linear-gradient(135deg,rgba(217,168,80,.12),rgba(120,92,168,.09));border:1px solid #27313d;border-radius:18px;padding:22px 24px;margin-bottom:8px}
.hero h2{margin:0 0 6px}.hero p{color:#8c97a5;margin:0;font-size:15px}
.stApp a{color:#d9a850}
.note{font-family:'Spectral';font-style:italic;color:#5c6675;font-size:13px}
div[data-testid="stHorizontalBlock"]{gap:14px}
/* tighter, friendlier tiles */
[data-testid="stImage"] img{border-radius:11px;border:1px solid #27313d}
.stButton button{border-radius:10px;border:1px solid #33404e;background:rgba(255,255,255,.02);color:#ece6d6;
  font-family:'IBM Plex Mono';font-size:12px;transition:.14s;padding:6px 10px}
.stButton button:hover{border-color:#d9a850;color:#f0d292}
.stButton button[kind="primary"]{background:linear-gradient(180deg,#f0d292,#d9a850);color:#160f04;border:none;font-family:'Cinzel';font-weight:600;letter-spacing:1px}
.tile{border:1px solid #27313d;border-radius:14px;padding:9px;background:linear-gradient(180deg,#141a21,#181f29);margin-bottom:6px}
.tile .nm{font-size:13px;font-weight:500;line-height:1.2;min-height:30px}
.badge{display:inline-block;font-family:'IBM Plex Mono';font-size:9px;letter-spacing:.5px;padding:2px 7px;border-radius:6px;border:1px solid #33404e;color:#8c97a5}
.badge.new{background:rgba(217,168,80,.15);color:#f0d292;border-color:rgba(217,168,80,.4)}
.badge.up{background:rgba(95,194,138,.13);color:#5fc28a;border-color:rgba(95,194,138,.35)}
.badge.dn{background:rgba(223,114,97,.13);color:#df7261;border-color:rgba(223,114,97,.35)}
.pwr{font-family:'IBM Plex Mono';font-weight:600;font-size:13px}
.statusbar{font-family:'IBM Plex Mono';font-size:11px;color:#8c97a5;letter-spacing:.4px;margin:2px 0 4px}
.statusbar .dot{color:#5fc28a}
.sheet{border:1px solid #33404e;border-radius:16px;padding:18px;background:linear-gradient(180deg,#141a21,#10151c);margin-bottom:14px}
.setline{border:1px solid #27313d;border-radius:11px;padding:9px 13px;margin-bottom:7px;background:rgba(255,255,255,.012);font-size:13.5px;display:flex;justify-content:space-between;align-items:center;gap:10px}
.muted{color:#8c97a5}
.note-row{border-left:2px solid #27313d;padding:4px 0 4px 12px;margin:6px 0;font-size:13.5px;line-height:1.5}
hr{border-color:#27313d}
/* uniform card art: fixed 4:3 crop so the grid is even, no weird shapes */
.cardart{width:100%;aspect-ratio:4/3;object-fit:cover;object-position:center 22%;border-radius:10px;
  border:1px solid #2b3542;display:block}
.tilewrap{margin-bottom:2px}
.tilewrap .meta{font-family:'IBM Plex Mono';font-size:11px;margin:5px 0 2px;display:flex;align-items:center;gap:7px;flex-wrap:wrap}
.tilewrap .tname{font-size:12.5px;line-height:1.25;height:32px;overflow:hidden;margin-top:4px}
/* shrink the gap Streamlit puts under images and buttons */
[data-testid="stImage"]{margin-bottom:2px}
.stButton{margin-top:0}
.element-container{margin-bottom:.35rem}
div[data-testid="stVerticalBlock"]{gap:.45rem}
/* metric cards look like a dashboard */
[data-testid="stMetric"]{background:linear-gradient(180deg,#141a21,#10151c);border:1px solid #27313d;
  border-radius:13px;padding:12px 14px;height:100%;display:flex;flex-direction:column;justify-content:flex-start}
/* keep every card in a row the same height even when one has a delta badge */
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div[data-testid="stColumn"]{display:flex}
div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) > div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"]{width:100%}
[data-testid="stMetricValue"]{font-size:1.7rem}
[data-testid="stMetricDelta"]{margin-top:2px}
[data-testid="stMetricLabel"]{font-family:'IBM Plex Mono';font-size:10px;letter-spacing:1px;text-transform:uppercase;color:#8c97a5}
/* dataframe a touch denser */
[data-testid="stDataFrame"]{border:1px solid #27313d;border-radius:12px}
/* suggestion buttons look like a dropdown list */
.sugg .stButton button{text-align:left;justify-content:flex-start;width:100%;background:rgba(255,255,255,.015);
  border:1px solid #222b35;font-family:'Spectral';font-size:14px;padding:9px 12px}
.sugg .stButton button:hover{background:rgba(217,168,80,.08)}
.legendrow{font-family:'IBM Plex Mono';font-size:11px;color:#8c97a5;margin:2px 0 10px;display:flex;gap:16px;flex-wrap:wrap}
.legendrow b{color:#ece6d6;font-weight:500}
.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:middle}
/* ---------- centered masthead ---------- */
.masthead{text-align:center;margin:6px auto 4px;max-width:760px}
.masthead .crest{font-size:40px;line-height:1;color:#d9a850;margin-bottom:2px;
  text-shadow:0 0 18px rgba(217,168,80,.45)}
.masthead .wordmark{font-family:'Cinzel',serif;font-weight:700;color:#f0d292;
  font-size:3.1rem;letter-spacing:.32em;text-indent:.32em;line-height:1.05;margin:0}
.masthead .subtitle{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:4px;
  color:#8c97a5;margin-top:8px}
.masthead .statusbar{justify-content:center;margin-top:8px}
.masthead .mast-divider{height:1px;width:200px;margin:16px auto 2px;
  background:linear-gradient(90deg,transparent,#d9a850,transparent);opacity:.6}
@media (max-width:640px){
  .masthead .wordmark{font-size:2rem;letter-spacing:.2em;text-indent:.2em}
  .masthead .crest{font-size:30px}
}
/* ---------- mobile (phones like iPhone SE, <= 640px) ---------- */
@media (max-width: 640px){
  .block-container{padding-left:.7rem!important;padding-right:.7rem!important}
  h1{font-size:1.9rem!important}
  h2{font-size:1.4rem!important}
  h3{font-size:1.15rem!important}
  /* most multi-column rows stack vertically so nothing becomes a sliver */
  div[data-testid="stHorizontalBlock"]{flex-wrap:wrap!important;gap:8px!important}
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]{
    flex:1 1 100%!important;width:100%!important;min-width:100%!important}
  /* card grids stay 2-up instead of 1-up so browsing is not endless scroll */
  div[data-testid="stColumn"]:has(img.cardart){flex:1 1 47%!important;min-width:47%!important;width:47%!important}
  [data-testid="stMetric"]{padding:9px 11px}
  [data-testid="stMetricValue"]{font-size:1.25rem!important}
  .tname{height:auto;font-size:12px}
  .stTabs [data-baseweb="tab"]{font-size:10.5px;padding:0 6px}
  .verdict{font-size:13.5px}
  .signalwrap svg{max-width:100%!important}
}
/* make the chart container never overflow horizontally */
[data-testid="stArrowVegaLiteChart"], .stVegaLiteChart{overflow-x:auto}
</style>
""", unsafe_allow_html=True)




# ============================================================================
# UI LAYER
# ============================================================================
ACCENT = "#d9a850"
SS = st.session_state
SS.setdefault("sel_card", None)
SS.setdefault("basket", [])
SS.setdefault("set_choice", None)
SS.setdefault("built_deck", None)


def score_color(v):
    return "#d9a850" if v >= 80 else "#9a7bc4" if v >= 65 else "#5294d6" if v >= 50 else "#7a8494"


# ---- header ----
with st.spinner("Opening the library..."):
    _hf_pool = load_full_pool_from_hf(HF_REPO)
    if _hf_pool and len(_hf_pool) > TARGET:
        POOL = [c for c in _hf_pool if c.get("name")]
        _pool_source = "hf"
    else:
        POOL = [c for c in load_library() if c.get("name")]
        _pool_source = "api"

if not POOL:
    st.error("Could not reach the live card service right now. Please reload the page in a moment.")
    st.stop()

# ---- train ML models on the scraped library (cached) ----
for _c in POOL:
    feats(_c)
_ml_names = [c["name"] for c in POOL]
_ml_rows = [ml_feature_row(c) for c in POOL]
_ml_prices = [price_now(c) for c in POOL]
_ml_ranks = [c.get("edhrec_rank") or 60000 for c in POOL]
_pool_sig = f"{len(POOL)}-{_ml_names[0] if _ml_names else ''}-{_ml_names[-1] if _ml_names else ''}"
ML, ML_NAMES = build_ml(_pool_sig, _ml_names, _ml_rows, _ml_prices, _ml_ranks)
ML_INDEX = {n: i for i, n in enumerate(ML_NAMES)}

# real accumulated price history from Hugging Face, if configured
HIST = load_price_history(HF_REPO)

# time-series forecaster: trains on HF history if enough data exists
FORECASTER = {"ok": False}
if HIST is not None and len(HIST) >= 80:
    _hist_rows_compact = [
        {"name": r["name"], "date": r["snapshot_date"], "usd": r["usd"]}
        for _, r in HIST.dropna(subset=["usd"]).iterrows()
    ]
    _attr_by_name = {c["name"]: ml_feature_row(c) for c in POOL}
    _hist_sig = f"{len(HIST)}-{HIST['snapshot_date'].min()}-{HIST['snapshot_date'].max()}"
    FORECASTER = train_forecaster(_hist_sig, _hist_rows_compact, _attr_by_name)

stamp = time.strftime("%I:%M %p").lstrip("0")
_pool_desc = (f"all {len(POOL):,} known cards" if _pool_source == "hf"
              else f"the {len(POOL):,} most-played cards")
st.markdown(
    "<div class='masthead'>"
    "<div class='crest'>✦</div>"
    "<div class='wordmark'>MANALORE</div>"
    "<div class='subtitle'>ACADEMY OF CARD MASTERY</div>"
    f"<div class='statusbar'><span class='dot'>●</span> Live market data, updated {stamp}"
    f"&nbsp;&nbsp;·&nbsp;&nbsp;tracking {_pool_desc}"
    + (f"&nbsp;&nbsp;·&nbsp;&nbsp;{HIST['snapshot_date'].nunique()} days of price history"
       if (HIST is not None and len(HIST)) else "")
    + "</div>"
    "<div class='mast-divider'></div>"
    "</div>",
    unsafe_allow_html=True)

tab_lib, tab_look, tab_sets, tab_build, tab_market, tab_academy = st.tabs(
    ["Library", "Lookup", "Sets", "Builder", "Market", "Academy"])


# ---- shared card sheet ----
def signal_bars_svg(f):
    """A compact, labeled bar readout (0-10) that scales to any width."""
    rows = [("Play-rate", f["play_rank"], "#d9a850"), ("Efficiency", f["eff"], "#5fc28a"),
            ("Card advantage", f["card_adv"], "#5294d6"), ("Flexibility", f["flex"], "#9a7bc4"),
            ("Format breadth", f["ubiq"], "#e0a64b"), ("Keywords", f["kw"], "#df7261")]
    h = len(rows) * 30 + 8
    W, lab_w, val_w = 400, 120, 30
    bar_max = W - lab_w - val_w - 8
    parts = [f"<div class='signalwrap'><svg viewBox='0 0 {W} {h}' width='100%' preserveAspectRatio='xMinYMin meet'>"]
    for i, (lab, val, col) in enumerate(rows):
        y = 10 + i * 30
        w = max(2, val / 10 * bar_max)
        parts.append(f"<text x='0' y='{y+11}' fill='#b9c2cf' font-family=\"IBM Plex Mono\" font-size='12'>{lab}</text>")
        parts.append(f"<rect x='{lab_w}' y='{y}' width='{bar_max}' height='14' rx='4' fill='#1a2128'/>")
        parts.append(f"<rect x='{lab_w}' y='{y}' width='{w:.0f}' height='14' rx='4' fill='{col}'/>")
        parts.append(f"<text x='{W-2}' y='{y+11}' text-anchor='end' fill='{col}' font-family=\"IBM Plex Mono\" font-size='12'>{val:.1f}</text>")
    parts.append("</svg></div>")
    return "".join(parts)


def card_sheet_body(c):
    """Card popup: image + verdict + metrics on top, then full-width charts. No empty gaps."""
    left, right = st.columns([1, 1.5], gap="medium")
    with left:
        im = img_uri(c, "normal") or img_uri(c, "large")
        if im:
            st.image(im, use_container_width=True)
        in_basket = c["name"] in [x["name"] for x in SS["basket"]]
        if st.button("✓ In your deck" if in_basket else "＋ Add to deck builder",
                     key=f"dlgadd_{c.get('id', c['name'])}", use_container_width=True,
                     type="secondary" if in_basket else "primary"):
            if not in_basket:
                SS["basket"].append(c)
                st.toast(f"Added {c['name']} to your deck builder.")
                st.rerun()
    with right:
        st.markdown(f"<span class='cap'>{type_line(c)} · {c.get('rarity','').title()}"
                    + (" · SUPPLY-LOCKED" if c.get("reserved") else "") + "</span>", unsafe_allow_html=True)
        st.markdown(f"<div class='verdict'><b class='gold'>In plain words</b><br>{plain_read(c)}</div>",
                    unsafe_allow_html=True)
        a, b, d = st.columns(3)
        a.metric("Power", importance(c))
        b.metric("Demand", demand(c))
        pj, p = proj_price(c), price_now(c)
        pval, psrc = price_info(c)
        delt = (pj / p * 100 - 100) if (pj and p) else 0
        d.metric("Price", fmt_usd(p) if p else "n/a", f"{delt:+.1f}% 90d" if p else None)
        if p and psrc != "market":
            st.markdown(f"<span class='cap'>Price shown is {psrc}; a standard market price is not listed.</span>",
                        unsafe_allow_html=True)
        elif not p:
            st.markdown("<span class='cap'>No market price is listed for this card yet.</span>",
                        unsafe_allow_html=True)
        st.markdown("**Signal strength**  <span class='cap'>what drives its power score, each out of 10</span>",
                    unsafe_allow_html=True)
        st.markdown(signal_bars_svg(feats(c)), unsafe_allow_html=True)
        fmts = legal_formats(c)
        st.markdown("<span class='cap'>LEGAL: " + (", ".join(f.title() for f in fmts) if fmts else "Limited")
                    + "</span>", unsafe_allow_html=True)

    # ---- real price history (from Hugging Face), if we have enough points ----
    hist_df = card_history(HIST, c["name"]) if HIST is not None else None
    if hist_df is not None and len(hist_df) >= 3:
        first, last = hist_df["usd"].iloc[0], hist_df["usd"].iloc[-1]
        chg = (last - first) / first * 100 if first else 0
        st.markdown("**Real price history**  "
                    f"<span class='cap'>{len(hist_df)} daily snapshots · "
                    f"{'up' if chg>=0 else 'down'} {abs(chg):.0f}% over the period</span>", unsafe_allow_html=True)
        try:
            import altair as alt
            hc = (alt.Chart(hist_df).mark_line(color="#d9a850", strokeWidth=2)
                  .encode(x=alt.X("snapshot_date:T", title=None),
                          y=alt.Y("usd:Q", title="USD"),
                          tooltip=[alt.Tooltip("snapshot_date:T", title="Date"),
                                   alt.Tooltip("usd:Q", title="Price", format="$.2f")])
                  .properties(height=190))
            st.altair_chart(hc, use_container_width=True)
        except Exception:
            st.line_chart(hist_df.set_index("snapshot_date")["usd"], color="#d9a850", height=190)

    if p:
        fc_result = forecast_30d(FORECASTER, c, HIST)
        mom = momentum_signal(HIST, c["name"]) if HIST is not None else None

        if fc_result:
            fc_pct, fc_conf = fc_result
            fc_col = "#5fc28a" if fc_pct >= 0 else "#df7261"
            st.markdown("**30-day price forecast**  "
                        f"<span class='cap'>trained on real price history, {fc_conf} confidence</span>",
                        unsafe_allow_html=True)
            f1, f2, f3 = st.columns(3)
            f1.metric("Forecast 30d", fmt_usd(p * (1 + fc_pct/100)) if p else "n/a",
                      f"{fc_pct:+.1f}%")
            if mom:
                f2.metric("7d momentum", f"{mom['mom_7d']*100:+.1f}%",
                          mom["trend"].title())
                f3.metric("14d volatility", f"{mom['vol_14d']*100:.1f}%",
                          mom["vol_lbl"].title())
        else:
            st.markdown("**Price outlook · next 90 days**  "
                        f"<span class='cap'>model projection, {confidence(c)}% confidence · "
                        f"forecaster trains once 30+ days of history accumulate</span>",
                        unsafe_allow_html=True)

        days = [0, 30, 60, 90]
        if fc_result:
            fc_pct30 = fc_result[0]
            series = [p, round(p * (1 + fc_pct30/100), 2),
                      round(p * (1 + fc_pct30/100 * 1.8), 2),
                      round(p * (1 + fc_pct30/100 * 2.4), 2)]
            chart_label = "Forecast USD"
            chart_color = "#5fc28a" if fc_pct30 >= 0 else "#df7261"
        else:
            series = [round(p * (1 + (d / 30) * drift_mo(c)), 2) for d in days]
            chart_label = "Projected USD"
            chart_color = "#5fc28a"

        out = pd.DataFrame({"Day": days, "Label": ["Now", "+30d", "+60d", "+90d"],
                            chart_label: series})
        try:
            import altair as alt
            lo, hi = min(series), max(series)
            pad = max(0.05, (hi - lo) * 0.4)
            line = (alt.Chart(out).mark_line(color=chart_color, strokeWidth=2.5,
                                             point=alt.OverlayMarkDef(color=chart_color))
                    .encode(x=alt.X("Day:Q", title=None, sort=None,
                                    scale=alt.Scale(domain=[0, 90]),
                                    axis=alt.Axis(values=days,
                                                  labelExpr="datum.value == 0 ? 'Now' : '+' + datum.value + 'd'")),
                            y=alt.Y(f"{chart_label}:Q", title="USD",
                                    scale=alt.Scale(domain=[max(0, lo - pad), hi + pad])),
                            tooltip=[alt.Tooltip("Label:N", title="When"),
                                     alt.Tooltip(f"{chart_label}:Q", format="$.2f")])
                    .properties(height=200))
            st.altair_chart(line, use_container_width=True)
        except Exception:
            st.line_chart(out.set_index("Label")[chart_label], color=chart_color, height=190)

        src = "Trained forecaster" if fc_result else "Model projection"
        st.caption(f"{src}. Supply risk {reprint_risk(c)}/10. Excludes surprise reprints and rules changes.")

    # ---- ML fair value ----
    sig = ml_value_signal(ML, c)
    if sig:
        fair, verdict, pct = sig
        vcol = {"overpriced": "#df7261", "underpriced": "#5fc28a", "fairly priced": "#d9a850"}[verdict]
        st.markdown("**ML fair value**  <span class='cap'>what a model trained on the whole library "
                    "expects this card to cost, from its traits</span>", unsafe_allow_html=True)
        fc1, fc2 = st.columns(2)
        fc1.metric("Model fair value", fmt_usd(fair))
        fc2.metric("Market vs model", f"{pct:+.0f}%", verdict)
        st.markdown(f"<span class='cap'>The market price is <b style='color:{vcol}'>{verdict}</b> "
                    f"relative to the model's estimate. This is fair-value analysis from one data snapshot, "
                    f"not a future-price forecast.</span>", unsafe_allow_html=True)

    st.markdown("**Analyst read**")
    label = {"strong": "🟢", "caution": "🔴", "note": "🟡"}
    for tag, txt in analyst_notes(c):
        st.markdown(f"<div class='note-row'>{label[tag]} &nbsp;{txt}</div>", unsafe_allow_html=True)

    # ---- origin + print history ----
    st.markdown("**Origin & printings**")
    o1, o2, o3 = st.columns(3)
    o1.metric("Original set", c.get("set_name", "Unknown"))
    o2.metric("Set code", (c.get("set", "") or "").upper())
    o3.metric("Released", c.get("released_at", "Unknown"))
    hist = print_history(c["name"])
    if hist and len(hist) > 1:
        st.markdown(f"<span class='cap'>This card has {len(hist)} printings. "
                    "Newer reprints usually ease supply and soften price.</span>", unsafe_allow_html=True)
        hdf = pd.DataFrame(hist)
        st.dataframe(hdf, use_container_width=True, hide_index=True, height=min(320, 60 + len(hist) * 35),
                     column_config={"Price": st.column_config.NumberColumn(format="$%.2f")})
    elif hist:
        st.markdown("<span class='cap'>Only one printing so far, so supply is concentrated in this release.</span>",
                    unsafe_allow_html=True)

    # ---- ML: cards that play similarly (learned feature space) ----
    if c["name"] in ML_INDEX:
        sim_idx = ml_similar(ML, ML_NAMES, ML_INDEX[c["name"]], k=6)
        sims = [POOL[i] for i in sim_idx]
        if sims:
            st.markdown("**Plays like this**  <span class='cap'>cards the model finds most similar, "
                        "useful as swaps or deck-mates</span>", unsafe_allow_html=True)
            card_tiles(sims, "mlsim", 6, 6)


@st.dialog(" ", width="large")
def open_card_dialog(c):
    st.markdown(f"## {c['name']}")
    card_sheet_body(c)


def show_card(c):
    open_card_dialog(c)


def card_tiles(cards, where, cols_n=5, limit=40):
    """Visual grid of card art with a uniform crop. Clicking opens the popup."""
    cards = cards[:limit]
    rows = (len(cards) + cols_n - 1) // cols_n
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_n)
        for col in cols:
            if idx >= len(cards):
                break
            c = cards[idx]; idx += 1
            with col:
                art = img_uri(c, "art_crop") or img_uri(c, "normal") or ""
                d = delta_pct(c)
                cls = "up" if d >= 0 else "dn"
                arrow = "▲" if d >= 0 else "▼"
                st.markdown(
                    f"<div class='tilewrap'>"
                    f"<img class='cardart' src='{art}' alt=''>"
                    f"<div class='meta'>"
                    f"<span class='pwr' style='color:{score_color(importance(c))}'>{importance(c)}</span>"
                    f"<span class='gold'>{fmt_usd(price_now(c))}</span>"
                    f"<span class='badge {cls}'>{arrow}{abs(d):.0f}%</span></div>"
                    f"<div class='tname'>{c['name']}</div></div>",
                    unsafe_allow_html=True)
                if st.button("View", key=f"{where}_open_{idx}_{c.get('id', c['name'])}", use_container_width=True):
                    show_card(c)


# ============================================================================
# LIBRARY
# ============================================================================
with tab_lib:
    _lib_desc = (f"Every known Magic card ({len(POOL):,} unique cards) from all sets and all time, "
                 if _pool_source == "hf" else
                 f"The {len(POOL):,} most-played cards, ")
    st.markdown(f"<div class='hero'><h2>Read any card like a master.</h2>"
                f"<p>{_lib_desc}each scored for power, demand and price outlook, "
                "in plain language for newcomers and full depth for veterans.</p></div>",
                unsafe_allow_html=True)

    avg_dem = int(np.mean([demand(c) for c in POOL]))
    priced_pool = [c for c in POOL if price_now(c)]
    gain = max(priced_pool, key=delta_pct) if priced_pool else POOL[0]
    gname = gain["name"].split(",")[0].split(" // ")[0]
    if len(gname) > 14:
        gname = gname[:13] + "…"
    reserved = sum(1 for c in POOL if c.get("reserved"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Demand", avg_dem)
    m2.metric("Top Gainer", gname, f"{delta_pct(gain):+.1f}% 90d")
    m3.metric("Supply-Locked", reserved)
    m4.metric("Cards Tracked", f"{len(POOL):,}")

    fc = st.columns([2, 2, 2])
    colors = fc[0].multiselect("Colors", ["W", "U", "B", "R", "G", "Colorless"], default=[])
    type_pick = fc[1].selectbox("Type", ["All", "Creature", "Instant", "Sorcery", "Artifact",
                                         "Enchantment", "Planeswalker", "Land"])
    sort_pick = fc[2].selectbox("Sort by", ["Power", "Demand", "Price: high to low",
                                           "Price: low to high", "Biggest movers", "Name A-Z"])
    name_q = st.text_input("Search within these cards", "", placeholder="Type part of a card name")

    def match_color(c):
        if not colors:
            return True
        ci = c.get("color_identity", [])
        if "Colorless" in colors and not ci:
            return True
        picked = [x for x in colors if x != "Colorless"]
        if not picked:
            return not ci
        return bool(ci) and all(x in picked for x in ci)

    flt = [c for c in POOL if match_color(c)
           and (type_pick == "All" or type_pick.lower() in type_line(c).lower())
           and (not name_q or name_q.lower() in c["name"].lower())]
    keyf = {"Power": lambda c: -importance(c), "Demand": lambda c: -demand(c),
            "Price: high to low": lambda c: -(price_now(c) or 0), "Price: low to high": lambda c: (price_now(c) or 0),
            "Biggest movers": lambda c: -abs(delta_pct(c)), "Name A-Z": lambda c: c["name"]}[sort_pick]
    flt.sort(key=keyf)

    if not flt:
        st.info("No cards match those filters. Try clearing a filter or widening the colors.")
    else:
        show_n = st.slider("How many to show", 20, 120, 60, 20, key="lib_show")
        st.markdown(f"<span class='cap'>{len(flt):,} cards match · showing {min(show_n, len(flt))}</span>",
                    unsafe_allow_html=True)
        card_tiles(flt, "lib", 5, show_n)


# ============================================================================
# LOOKUP
# ============================================================================
with tab_look:
    st.markdown("### Find a card")
    st.markdown("<span class='cap'>Start typing. Suggestions appear instantly, like a search engine. "
                "Partial names and typos are fine.</span>", unsafe_allow_html=True)

    q = st.text_input("Card name", "", placeholder="e.g. spider, bolt, ragavan, miles morales",
                      key="look_q")
    if q and len(q) >= 2:
        names = scry_autocomplete(q)
        if names:
            st.markdown(f"<span class='cap'>{len(names)} suggestion(s) · tap to open</span>", unsafe_allow_html=True)
            st.markdown("<div class='sugg'>", unsafe_allow_html=True)
            for i, n in enumerate(names[:10]):
                if st.button(f"🔍  {n}", key=f"sugg_{i}", use_container_width=True):
                    fb = scry_named(n)
                    if fb:
                        show_card(fb)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            fb = scry_named(q)
            if fb and fb.get("name"):
                st.markdown("<span class='cap'>Closest match</span>", unsafe_allow_html=True)
                st.markdown("<div class='sugg'>", unsafe_allow_html=True)
                if st.button(f"🔍  {fb['name']}", key="sugg_fb", use_container_width=True):
                    show_card(fb)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.info("No card found for that yet. Check the spelling, or try fewer letters.")
    else:
        st.markdown("<span class='cap'>Popular searches:</span>", unsafe_allow_html=True)
        chips = ["Ragavan", "Sol Ring", "Counterspell", "Spider-Man", "Lightning Bolt"]
        cc = st.columns(len(chips))
        for i, ch in enumerate(chips):
            if cc[i].button(ch, key=f"chip_{i}", use_container_width=True):
                fb = scry_named(ch)
                if fb:
                    show_card(fb)


# ============================================================================
# SETS
# ============================================================================
with tab_sets:
    st.markdown("### Sets & Releases")
    st.markdown("<span class='cap'>Every paper release, newest first. Search a series like Spider-Man or Final Fantasy.</span>",
                unsafe_allow_html=True)
    sets = load_sets()
    if not sets:
        st.info("Could not load the set list right now. Please reload in a moment.")
    else:
        def bucket(s):
            t = s.get("set_type")
            if t in ("expansion", "core"): return "Expansion"
            if t == "commander": return "Commander"
            if t in ("masters", "draft_innovation"): return "Masters"
            return "Other"

        today = time.strftime("%Y-%m-%d")

        def newness(s):
            ra = s.get("released_at", "")
            if ra and ra > today:
                return "<span class='badge new'>UPCOMING</span>"
            if ra:
                try:
                    days = (time.time() - time.mktime(time.strptime(ra, "%Y-%m-%d"))) / 86400
                    if 0 <= days < 95:
                        return "<span class='badge new'>NEW</span>"
                except ValueError:
                    pass
            return ""

        if SS.get("set_choice"):
            choice = SS["set_choice"]
            if st.button("← All sets", key="set_back"):
                SS["set_choice"] = None
                st.rerun()
            st.markdown(f"#### {choice['name']} {newness(choice)}", unsafe_allow_html=True)
            st.markdown(f"<span class='cap'>{choice.get('code','').upper()} · released "
                        f"{choice.get('released_at','TBA')} · {choice['card_count']} cards in set</span>",
                        unsafe_allow_html=True)
            code = choice["code"]
            with st.spinner("Loading every card in this set..."):
                all_cards = load_full_set(code, unique="prints")
            if not all_cards:
                st.info("Could not load this set right now. Please reload in a moment.")
            else:
                fr1 = st.columns([2, 1, 1, 1])
                name_in = fr1[0].text_input("Search this set", "", key="setcard_q",
                                            placeholder="Type a card name...")
                finish = fr1[1].selectbox("Finish", ["Any", "Foil", "Non-foil"], key="set_finish")
                booster = fr1[2].selectbox("Booster", ["All", "Play Booster", "Collector Booster"], key="set_boost",
                                           help="Play Booster shows standard versions; Collector shows special treatments.")
                sort_by = fr1[3].selectbox("Sort", ["Most played", "Price high", "Price low", "Name", "Mana value"],
                                           key="set_sort")
                fr2 = st.columns(5)
                rarities = sorted({c.get("rarity", "").title() for c in all_cards if c.get("rarity")})
                rarity_pick = fr2[0].multiselect("Rarity", rarities, key="set_rarity")
                color_pick = fr2[1].multiselect("Color", ["White", "Blue", "Black", "Red", "Green", "Multicolor", "Colorless"],
                                                key="set_color")
                type_pick = fr2[2].multiselect("Type", ["Creature", "Instant", "Sorcery", "Artifact",
                                                        "Enchantment", "Planeswalker", "Land", "Battle"], key="set_type2")
                mvs = sorted({min(8, int(c.get("cmc", 0))) for c in all_cards})
                mv_pick = fr2[3].multiselect("Mana value", [(f"{m}" if m < 8 else "8+") for m in mvs], key="set_mv")
                legal_pick = fr2[4].multiselect("Legal in", [f.title() for f in FMTS], key="set_legal")

                def keep(c):
                    if name_in and name_in.lower() not in c["name"].lower():
                        return False
                    if finish == "Foil" and not has_foil(c):
                        return False
                    if finish == "Non-foil" and not has_nonfoil(c):
                        return False
                    if booster == "Play Booster" and treatment(c) != "Standard":
                        return False
                    if booster == "Collector Booster" and treatment(c) == "Standard":
                        return False
                    if rarity_pick and c.get("rarity", "").title() not in rarity_pick:
                        return False
                    if color_pick and color_category(c) not in color_pick:
                        return False
                    if type_pick and primary_type(c) not in type_pick:
                        return False
                    if mv_pick:
                        mv = min(8, int(c.get("cmc", 0)))
                        if (f"{mv}" if mv < 8 else "8+") not in mv_pick:
                            return False
                    if legal_pick:
                        lf = {f.title() for f in legal_formats(c)}
                        if not any(lp in lf for lp in legal_pick):
                            return False
                    return True

                filtered = [c for c in all_cards if keep(c)]
                sort_key = {
                    "Most played": lambda c: -importance(c),
                    "Price high": lambda c: -(price_now(c) or 0),
                    "Price low": lambda c: (price_now(c) or 1e9),
                    "Name": lambda c: c["name"],
                    "Mana value": lambda c: c.get("cmc", 0),
                }[sort_by]
                filtered.sort(key=sort_key)

                active = any([name_in, finish != "Any", booster != "All", rarity_pick, color_pick,
                              type_pick, mv_pick, legal_pick])
                st.markdown(f"<span class='cap'>{len(filtered)} of {len(all_cards)} printings"
                            f"{' · filtered' if active else ' · showing all'}</span>", unsafe_allow_html=True)
                if not filtered:
                    st.info("No cards match these filters. Clear a filter to see more.")
                else:
                    show_n = st.slider("How many to show", 20, 200, 60, 20, key="set_show")
                    card_tiles(filtered, "set", 5, show_n)
        else:
            c1, c2 = st.columns([1, 2])
            set_type = c1.selectbox("Category", ["All", "Expansion", "Commander", "Masters", "Other"])
            set_q = c2.text_input("Find a set or series", "",
                                  placeholder="e.g. Spider-Man, Final Fantasy, Lorwyn, Avatar")
            shown = [s for s in sets if (set_type == "All" or bucket(s) == set_type)]
            if set_q:
                ql = set_q.lower()
                exact = [s for s in shown if ql in s["name"].lower() or ql in s.get("code", "").lower()]
                if not exact:
                    # forgiving: match any 3+ letter fragment of any typed word
                    frags = [w for w in ql.split() if len(w) >= 3]
                    exact = [s for s in shown if any(fr in s["name"].lower() for fr in frags)]
                shown = exact
            if not shown:
                st.info(f"No sets match \u201c{set_q}\u201d. Try a shorter word, like \u201cspider\u201d or \u201cfinal\u201d.")
            else:
                st.markdown(f"<span class='cap'>{len(shown)} set(s) · newest first</span>", unsafe_allow_html=True)
                for i, s in enumerate(shown[:60]):
                    cL, cR = st.columns([5, 1])
                    with cL:
                        st.markdown(f"<div class='setline'><span>{s['name']} {newness(s)}"
                                    f"<br><span class='cap'>{s.get('code','').upper()} · {s.get('released_at','TBA')} "
                                    f"· {s['card_count']} cards · {bucket(s)}</span></span></div>",
                                    unsafe_allow_html=True)
                    with cR:
                        if st.button("Open", key=f"setopen_{i}_{s['code']}", use_container_width=True):
                            SS["set_choice"] = s
                            st.rerun()


# ============================================================================
# BUILDER
# ============================================================================
with tab_build:
    st.markdown("### Build a deck")
    st.markdown("<span class='cap'>Tell us the brief, optionally tap a few favourite cards, and we assemble "
                "a synergy-first deck from across all of Magic, grade it, and let you download it.</span>",
                unsafe_allow_html=True)

    st.markdown("**Step 1 · Your brief**  <span class='cap'>this guides the engine</span>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    purpose = b1.selectbox("Purpose", ["Casual", "Competitive", "Budget"],
                           help="Budget keeps cards cheap. Competitive favours format staples.")
    strategy = b2.selectbox("Strategy", ["Aggro", "Midrange", "Control", "Combo"],
                            help="How the deck wins: fast creatures, value, answers, or a combo engine.")
    b3, b4 = st.columns(2)
    fmt = b3.selectbox("Format", ["Commander", "Modern", "Standard", "Pioneer", "Legacy", "Pauper", "Any"])
    color_pick = b4.multiselect("Colors", ["W", "U", "B", "R", "G"], default=[],
                                help="Leave empty and we infer colors from your chosen cards.")
    key_card = st.text_input("Build around a key card or commander (optional)",
                             placeholder="e.g. Atraxa, Grand Unifier")

    deck_size = 100 if fmt == "Commander" else 60
    lands_target = deck_lands(strategy, deck_size)
    spell_slots = deck_size - lands_target

    # ---- requirements panel (always visible) ----
    with st.container():
        st.markdown("**Deck requirements**  "
                    f"<span class='cap'>{fmt} · {strategy}</span>", unsafe_allow_html=True)
        rq = st.columns(4)
        rq[0].metric("Deck size", deck_size, "singleton" if deck_size >= 100 else "up to 4 copies")
        rq[1].metric("Lands", lands_target)
        rq[2].metric("Spells", spell_slots)
        tgts = role_targets(strategy, spell_slots)
        rq[3].metric("Core roles", len(tgts))
        rec = " · ".join(f"{r} ~{n}" for r, n in sorted(tgts.items(), key=lambda kv: -kv[1]))
        st.markdown(f"<span class='cap'>Recommended mix: {rec}</span>", unsafe_allow_html=True)

    # ---- favourites (persistent, no reset) ----
    st.markdown("**Step 2 · Add a few favourites (optional)**", unsafe_allow_html=True)
    picked_n = len(SS["basket"])
    prog = min(1.0, picked_n / max(1, spell_slots))
    st.progress(prog, text=f"{picked_n} favourite(s) chosen · we will fill the remaining "
                           f"{max(0, spell_slots - picked_n)} spell slots for you")
    with st.expander("Browse popular cards to add", expanded=False):
        suggest_pool = sorted(POOL, key=lambda c: -importance(c))[:30]
        basket_names = [b["name"] for b in SS["basket"]]
        idx = 0
        for _ in range((len(suggest_pool) + 4) // 5):
            cols_ = st.columns(5)
            for col in cols_:
                if idx >= len(suggest_pool):
                    break
                c = suggest_pool[idx]; idx += 1
                with col:
                    art = img_uri(c, "art_crop") or img_uri(c, "normal") or ""
                    st.markdown(f"<img class='cardart' src='{art}' alt=''>", unsafe_allow_html=True)
                    chosen = c["name"] in basket_names
                    st.markdown(f"<div class='tname'>{c['name']}</div>", unsafe_allow_html=True)
                    if st.button("✓ Added" if chosen else "＋ Add", key=f"pick_{idx}_{c.get('id', c['name'])}",
                                 use_container_width=True, type="primary" if chosen else "secondary"):
                        if chosen:
                            SS["basket"] = [b for b in SS["basket"] if b["name"] != c["name"]]
                        else:
                            SS["basket"].append(c)
                        st.rerun()

    if SS["basket"]:
        st.markdown(f"**Your favourites · {len(SS['basket'])}**")
        bcols = st.columns(5)
        for i, b in enumerate(SS["basket"]):
            with bcols[i % 5]:
                if st.button(f"✕ {b['name'][:16]}", key=f"rm_{i}_{b.get('id', b['name'])}", use_container_width=True):
                    SS["basket"].pop(i)
                    st.rerun()

    # ---- build trigger ----
    st.markdown("**Step 3 · Build**")
    cbuild = st.columns([3, 1])
    build = cbuild[0].button(f"⚔  Build my {fmt} deck", use_container_width=True, type="primary")
    if cbuild[1].button("Clear all", use_container_width=True):
        SS["basket"] = []
        SS["built_deck"] = None
        st.rerun()

    if build:
        seed = scry_named(key_card) if key_card.strip() else None
        starters = list(SS["basket"])
        if seed and seed.get("name") and seed["name"] not in [s["name"] for s in starters]:
            feats(seed); starters.insert(0, seed)
        cols = list(color_pick)
        if not cols:
            cs = set()
            for c in starters:
                cs.update(c.get("color_identity", []))
            cols = [x for x in ["W", "U", "B", "R", "G"] if x in cs]
        colors_key = "".join(sorted(cols))
        with st.spinner(f"Searching all of Magic, then using the synergy model to assemble a {strategy.lower()} deck..."):
            candidates = [enrich_card(c) for c in build_candidate_pool(colors_key, strategy, fmt, purpose)]
            used, lands, spell_slots2, targets = assemble_deck(starters, candidates, strategy, deck_size, models=ML)
            if len(used) < spell_slots2:  # top-up from library if the search was thin
                names = {c["name"] for c in used}
                for c in sorted(POOL, key=lambda c: -importance(c)):
                    if len(used) >= spell_slots2:
                        break
                    if c["name"] not in names and "land" not in type_line(c).lower():
                        used.append(c); names.add(c["name"])
            if not cols:
                cs = set()
                for c in used:
                    cs.update(c.get("color_identity", []))
                cols = [x for x in ["W", "U", "B", "R", "G"] if x in cs]
        # store everything in state so popups/reruns never lose the deck
        SS["built_deck"] = {
            "used": used, "lands": lands, "targets": targets, "strategy": strategy, "fmt": fmt,
            "deck_size": deck_size, "cols": cols, "purpose": purpose, "color_pick": color_pick,
            "seed_name": seed["name"] if seed else None, "fav_n": len(SS["basket"]),
        }

    # ---- render the built deck from state (survives reruns) ----
    bd = SS.get("built_deck")
    if bd:
        used, lands, targets = bd["used"], bd["lands"], bd["targets"]
        strategy, fmt, deck_size, cols = bd["strategy"], bd["fmt"], bd["deck_size"], bd["cols"]
        purpose = bd["purpose"]
        if not used:
            st.warning("We could not find cards for that brief. Try widening colors or strategy.")
        else:
            name = deck_name(cols, strategy)
            priced_cards = [c for c in used if price_now(c)]
            value = sum(price_now(c) for c in priced_cards)
            unpriced = len(used) - len(priced_cards)
            # brief match score: roles covered + color fit + format
            from collections import Counter
            rc = Counter(("Other" if role(c) == "Spell" else role(c)) for c in used)
            covered = sum(1 for r in targets if rc.get(r, 0) >= max(1, int(targets[r] * 0.5)))
            brief_match = clamp(covered / max(1, len(targets)) * 100, 0, 100)
            q = deck_quality(used, strategy, deck_size, value, brief_match, models=ML)

            st.markdown(f"## {name}")
            seed_txt = f"built around {bd['seed_name']}, " if bd.get("seed_name") else ""
            fav_txt = f"your {bd['fav_n']} favourite(s), " if bd.get("fav_n") else ""
            ml_txt = "the synergy model picked cards that cluster around your build, then " if (ML.get("ok") and "scaler" in ML) else ""
            st.markdown(f"<span class='cap'>A {purpose.lower()} {strategy.lower()} {fmt} deck, {seed_txt}{fav_txt}"
                        f"assembled from across all of Magic: {ml_txt}tuned to {guild_name(cols)}. "
                        f"{STRAT_PROFILE.get(strategy, {}).get('tagline','')}.</span>", unsafe_allow_html=True)

            # ---- quality scorecard ----
            st.markdown("#### Deck quality")
            qc = st.columns(6)
            qc[0].metric("Overall", q["overall"])
            qc[1].metric("Synergy", q["synergy"])
            qc[2].metric("Consistency", q["consistency"])
            qc[3].metric("Avg power", q["power"])
            qc[4].metric("Budget eff.", q["budget_eff"])
            qc[5].metric("Brief match", q["alignment"])
            grade = ("Tournament-ready" if q["overall"] >= 78 else "Strong casual" if q["overall"] >= 64
                     else "Fun, needs tuning" if q["overall"] >= 50 else "Rough draft")
            syn_note = (" Synergy uses the ML model's measure of how tightly the cards cluster together."
                        if (ML.get("ok") and "scaler" in ML) else "")
            st.markdown(f"<span class='cap'>Verdict: <b class='gold'>{grade}</b>. "
                        f"Scores blend synergy, consistency, power, budget efficiency and brief match.{syn_note}</span>",
                        unsafe_allow_html=True)

            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Deck size", len(used) + lands, f"{len(used)} spells / {lands} lands")
            mm2.metric("Est. value", f"${value:.0f}", budget_label(value))
            mm3.metric("Colors", guild_name(cols))
            mm4.metric("Win speed", {"Aggro": "Fast", "Midrange": "Medium", "Control": "Slow", "Combo": "Explosive"}[strategy])
            if unpriced:
                st.markdown(f"<span class='cap'>Value covers the {len(priced_cards)} cards with a listed price; "
                            f"{unpriced} had no price and lands are not counted.</span>", unsafe_allow_html=True)

            # ---- validation ----
            st.markdown("#### Validation  <span class='cap'>does the deck match your brief?</span>", unsafe_allow_html=True)
            for ok, msg in validate_deck(used, lands, targets, strategy, deck_size, cols, bd["color_pick"]):
                st.markdown(f"<div class='note-row'>{'🟢' if ok else '🔴'} &nbsp;{msg}</div>", unsafe_allow_html=True)

            # ---- charts ----
            ch1, ch2 = st.columns(2)
            try:
                import altair as alt
                with ch1:
                    st.markdown("**Mana curve**  <span class='cap'>cards at each cost</span>", unsafe_allow_html=True)
                    cv = {}
                    for c in used:
                        k = min(7, int(c.get("cmc", 0))); cv[k] = cv.get(k, 0) + 1
                    mv_order = [(f"{k}" if k < 7 else "7+") for k in range(8)]
                    cvdf = pd.DataFrame({"Mana value": mv_order, "Cards": [cv.get(k, 0) for k in range(8)]})
                    base = alt.Chart(cvdf).encode(
                        x=alt.X("Mana value:N", sort=mv_order, title="Mana value",
                                axis=alt.Axis(labelAngle=0)),
                        y=alt.Y("Cards:Q", title="Cards"))
                    bars = base.mark_bar(color=ACCENT, cornerRadius=3).encode(tooltip=["Mana value", "Cards"])
                    labels = base.mark_text(dy=-6, color="#ece6d6", fontSize=11).encode(
                        text=alt.condition("datum.Cards > 0", "Cards:Q", alt.value("")))
                    st.altair_chart((bars + labels).properties(height=230), use_container_width=True)
                with ch2:
                    st.markdown("**Color split**  <span class='cap'>mana symbols in the deck</span>", unsafe_allow_html=True)
                    pip = {"W": 0, "U": 0, "B": 0, "R": 0, "G": 0}
                    for c in used:
                        for s in re.findall(r"\{([WUBRG])\}", c.get("mana_cost", "") or ""):
                            pip[s] += 1
                    cmap = {"W": "#f3ecd6", "U": "#5294d6", "B": "#9a7bc4", "R": "#d65a48", "G": "#56a96e"}
                    pdf = pd.DataFrame({"Color": [k for k in pip if pip[k]], "Pips": [pip[k] for k in pip if pip[k]]})
                    if not pdf.empty:
                        st.altair_chart((alt.Chart(pdf).mark_arc(innerRadius=45)
                                         .encode(theta="Pips:Q",
                                                 color=alt.Color("Color:N", scale=alt.Scale(
                                                     domain=list(cmap.keys()), range=list(cmap.values())), legend=None),
                                                 tooltip=["Color", "Pips"]).properties(height=230)),
                                        use_container_width=True)
                    else:
                        st.caption("Colorless deck.")
                st.markdown("**Role mix**  <span class='cap'>what jobs the cards do</span>", unsafe_allow_html=True)
                rdf = pd.DataFrame({"Role": list(rc.keys()), "Cards": list(rc.values())})
                rbase = alt.Chart(rdf).encode(y=alt.Y("Role:N", sort="-x", title=None), x=alt.X("Cards:Q", title="Cards"))
                rbars = rbase.mark_bar(color="#9a7bc4", cornerRadius=3).encode(tooltip=["Role", "Cards"])
                rlabels = rbase.mark_text(dx=8, color="#ece6d6", fontSize=11).encode(text="Cards:Q")
                st.altair_chart((rbars + rlabels).properties(height=220), use_container_width=True)
            except Exception:
                with ch1:
                    st.markdown("**Mana curve**")
                    st.bar_chart(pd.Series([min(7, int(c.get("cmc", 0))) for c in used]).value_counts().sort_index(),
                                 color=ACCENT, height=220)
                with ch2:
                    st.markdown("**Role mix**")
                    st.bar_chart(pd.Series(dict(rc)).sort_values(ascending=False), color="#9a7bc4", height=220)

            # ---- gameplay guide ----
            with st.expander("📖  How to play this deck", expanded=False):
                st.markdown(f"**Win condition:** {win_condition(used, strategy)}")
                _seen, _stars = set(), []
                for c in sorted(used, key=lambda c: -importance(c)):
                    if c["name"] not in _seen:
                        _seen.add(c["name"]); _stars.append(c)
                    if len(_stars) >= 4:
                        break
                st.markdown("**Key cards & synergies:** Your highest-impact pieces are "
                            + ", ".join(c["name"] for c in _stars)
                            + ". Build your turns around resolving and protecting these.")
                mull = {"Aggro": "Keep hands with 2-3 lands and at least two cheap threats. Mulligan slow, land-light, or no-pressure hands.",
                        "Midrange": "Keep 3-4 lands with a mix of early plays and a payoff. Mulligan one-landers and hands with no early interaction.",
                        "Control": "Keep 3-5 lands with early removal or card draw. Mulligan hands with no early answers.",
                        "Combo": "Keep hands that progress toward your engine with some protection. Mulligan hands with no combo pieces or no mana."}[strategy]
                st.markdown(f"**Mulligan:** {mull}")
                plans = {"Aggro": ("Deploy threats every turn and attack.", "Push damage, use removal only to clear blockers.", "Burn or swing for lethal; empty your hand."),
                         "Midrange": ("Develop mana and trade with early threats.", "Deploy your best threats and grind card advantage.", "Close with your strongest cards once ahead."),
                         "Control": ("Survive: remove early threats, hit land drops.", "Trade one-for-one and start drawing extra cards.", "Land a finisher and protect it to close."),
                         "Combo": ("Set up mana and dig for pieces.", "Assemble and protect the engine.", "Execute the combo for the win.")}[strategy]
                st.markdown(f"**Early game:** {plans[0]}")
                st.markdown(f"**Mid game:** {plans[1]}")
                st.markdown(f"**Late game:** {plans[2]}")

            # ---- downloads ----
            brief_line = f"A {purpose.lower()} {strategy.lower()} {fmt} deck in {guild_name(cols)}."
            dl1, dl2 = st.columns(2)
            dl1.download_button("⬇  Decklist (CSV)", decklist_csv(used, lands, cols),
                                file_name=f"{name.replace(' ', '_')}.csv", mime="text/csv", use_container_width=True)
            dl2.download_button("⬇  Analysis report (TXT)",
                                deck_report_txt(name, brief_line, q, used, lands, value, cols, strategy, deck_size),
                                file_name=f"{name.replace(' ', '_')}_report.txt", mime="text/plain", use_container_width=True)

            # ---- visual deck grid (tap to inspect; basket persists) ----
            st.markdown("**The deck**  <span class='cap'>tap any card to inspect it</span>", unsafe_allow_html=True)
            by_role = {}
            for c in used:
                r = "Other" if role(c) == "Spell" else role(c)
                by_role.setdefault(r, []).append(c)
            order = ["Threat", "Removal", "Counter", "Card Advantage", "Disruption", "Ramp", "Engine", "Other"]
            for r in [x for x in order if x in by_role]:
                from collections import Counter as _C
                cnt = _C(c["name"] for c in by_role[r])
                uniq, seen = [], set()
                for c in sorted(by_role[r], key=lambda c: -importance(c)):
                    if c["name"] not in seen:
                        seen.add(c["name"]); uniq.append(c)
                st.markdown(f"**{r} · {len(by_role[r])}**  "
                            f"<span class='cap'>{len(uniq)} unique</span>", unsafe_allow_html=True)
                card_tiles(uniq, f"deck_{r}", 6, 60)
            st.markdown(f"**Lands · {lands}**  <span class='cap'>tuned to your colors ({''.join(cols) or 'C'})</span>",
                        unsafe_allow_html=True)


# ============================================================================
# MARKET
# ============================================================================
with tab_market:
    st.markdown("### Market")
    st.markdown("<span class='cap'>How the tracked cards are trending, and where value sits. "
                "Movement is a 90-day model projection, not a guarantee.</span>", unsafe_allow_html=True)

    priced = [c for c in POOL if price_now(c)]
    bull = sum(1 for c in priced if delta_pct(c) > 0.3)
    bear = sum(1 for c in priced if delta_pct(c) < -0.3)
    flat = len(priced) - bull - bear
    total_val = sum(price_now(c) for c in priced)
    hi = max(priced, key=lambda c: price_now(c)) if priced else None
    mk1, mk2, mk3, mk4 = st.columns(4)
    mk1.metric("Trending up", bull)
    mk2.metric("Trending down", bear)
    mk3.metric("Holding steady", flat)
    mk4.metric("Total tracked value", fmt_usd(total_val))
    if hi:
        st.markdown(f"<span class='cap'>Priciest card tracked: <b class='gold'>{hi['name']}</b> at "
                    f"{fmt_usd(price_now(hi))}. Prices are aggregated market values for study, not a quote.</span>",
                    unsafe_allow_html=True)

    # build a clean dataframe once
    rows = []
    for c in priced:
        p = price_now(c)
        rows.append({"Card": c["name"], "Power": importance(c), "Demand": demand(c),
                     "Price": round(p, 2), "Proj 90d": round(proj_price(c), 2),
                     "Move %": round(delta_pct(c), 1), "Confidence": confidence(c),
                     "Role": role(c)})
    df = pd.DataFrame(rows)

    st.divider()
    st.markdown("#### Demand vs price")
    st.markdown("<div class='legendrow'>Each dot is a card. Bigger dot means higher power. "
                "<span><span class='swatch' style='background:#5fc28a'></span>projected to rise</span>"
                "<span><span class='swatch' style='background:#df7261'></span>projected to fall</span> "
                "Right means more in demand; higher means pricier.</div>", unsafe_allow_html=True)
    try:
        import altair as alt
        sc = df[df["Price"] > 0].copy()
        sc["Trend"] = np.where(sc["Move %"] >= 0, "Rising", "Falling")
        chart = (alt.Chart(sc).mark_circle(opacity=0.65)
                 .encode(
                     x=alt.X("Demand:Q", title="Demand index (higher = more played)",
                             scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(grid=True)),
                     y=alt.Y("Price:Q", title="Market price, USD (log scale)",
                             scale=alt.Scale(type="log"), axis=alt.Axis(grid=True)),
                     size=alt.Size("Power:Q", scale=alt.Scale(range=[25, 360]), title="Power"),
                     color=alt.Color("Trend:N", scale=alt.Scale(domain=["Rising", "Falling"],
                                                                range=["#5fc28a", "#df7261"]), legend=None),
                     tooltip=["Card", "Power", "Demand", alt.Tooltip("Price:Q", format="$.2f"), "Move %"])
                 .properties(height=340))
        st.altair_chart(chart, use_container_width=True)
        st.caption("A log price scale spreads cheap and expensive cards evenly so the pattern is readable.")
    except Exception:
        st.scatter_chart(df, x="Demand", y="Price", height=320)

    # ---- two more charts for richer info ----
    st.divider()
    ec1, ec2 = st.columns(2)
    try:
        import altair as alt
        with ec1:
            st.markdown("**Price distribution**  <span class='cap'>how many cards in each price band</span>",
                        unsafe_allow_html=True)
            def band(p):
                return ("Under $1" if p < 1 else "$1-5" if p < 5 else "$5-20" if p < 20
                        else "$20-100" if p < 100 else "Over $100")
            order = ["Under $1", "$1-5", "$5-20", "$20-100", "Over $100"]
            bd_counts = {b: 0 for b in order}
            for c in priced:
                bd_counts[band(price_now(c))] += 1
            bdf = pd.DataFrame({"Band": order, "Cards": [bd_counts[b] for b in order]})
            bbase = alt.Chart(bdf).encode(x=alt.X("Band:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
                                          y=alt.Y("Cards:Q", title="Cards"))
            st.altair_chart((bbase.mark_bar(color="#5294d6", cornerRadius=3).encode(tooltip=["Band", "Cards"])
                             + bbase.mark_text(dy=-6, color="#ece6d6", fontSize=11).encode(
                                 text=alt.condition("datum.Cards > 0", "Cards:Q", alt.value("")))
                             ).properties(height=260), use_container_width=True)
        with ec2:
            st.markdown("**Average price by role**  <span class='cap'>which card jobs cost the most</span>",
                        unsafe_allow_html=True)
            role_prices = {}
            for c in priced:
                role_prices.setdefault(role(c), []).append(price_now(c))
            rpdf = pd.DataFrame({"Role": list(role_prices.keys()),
                                 "Avg price": [round(np.mean(v), 2) for v in role_prices.values()]})
            rpbase = alt.Chart(rpdf).encode(y=alt.Y("Role:N", sort="-x", title=None),
                                            x=alt.X("Avg price:Q", title="Average price (USD)"))
            st.altair_chart((rpbase.mark_bar(color="#d9a850", cornerRadius=3).encode(
                                tooltip=["Role", alt.Tooltip("Avg price:Q", format="$.2f")])
                             + rpbase.mark_text(dx=8, color="#ece6d6", fontSize=11).encode(
                                 text=alt.Tooltip("Avg price:Q", format="$.0f"))
                             ).properties(height=260), use_container_width=True)
    except Exception:
        pass

    st.divider()
    st.markdown("#### Biggest movers · next 90 days")
    up_col, dn_col = st.columns(2)
    risers = df.sort_values("Move %", ascending=False).head(8)
    fallers = df.sort_values("Move %").head(8)
    with up_col:
        st.markdown("<span class='cap up'>▲ Projected to rise</span>", unsafe_allow_html=True)
        st.dataframe(risers[["Card", "Price", "Proj 90d", "Move %"]], use_container_width=True, hide_index=True,
                     column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                                    "Proj 90d": st.column_config.NumberColumn(format="$%.2f"),
                                    "Move %": st.column_config.NumberColumn(format="%+.1f%%")})
    with dn_col:
        st.markdown("<span class='cap dn'>▼ Projected to dip</span>", unsafe_allow_html=True)
        st.dataframe(fallers[["Card", "Price", "Proj 90d", "Move %"]], use_container_width=True, hide_index=True,
                     column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                                    "Proj 90d": st.column_config.NumberColumn(format="$%.2f"),
                                    "Move %": st.column_config.NumberColumn(format="%+.1f%%")})

    st.divider()
    st.markdown("#### Full valuation table")
    st.markdown("<span class='cap'>Sort any column. Power and demand are model scores; price is the live market value.</span>",
                unsafe_allow_html=True)
    st.dataframe(df.sort_values("Power", ascending=False), use_container_width=True, hide_index=True, height=440,
                 column_config={"Price": st.column_config.NumberColumn(format="$%.2f"),
                                "Proj 90d": st.column_config.NumberColumn(format="$%.2f"),
                                "Move %": st.column_config.NumberColumn(format="%+.1f%%"),
                                "Power": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d"),
                                "Demand": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")})


# ============================================================================
# ACADEMY
# ============================================================================
with tab_academy:
    st.markdown("### The Academy")
    st.markdown("<span class='cap'>Learn how to read a card, from first lesson to deep theory.</span>",
                unsafe_allow_html=True)
    st.markdown("#### First lessons")
    l1, l2, l3, l4 = st.columns(4)
    l1.markdown("**Power**\n\nA 0 to 100 score. Higher means the card shapes games and shows up in real decks. 80+ is a cornerstone.")
    l2.markdown("**Demand**\n\nHow wanted a card is now, driven by how often it gets played. High demand supports a higher price.")
    l3.markdown("**90d Outlook**\n\nWhere the price could drift over three months if demand and supply hold. A guide, not a guarantee.")
    l4.markdown("**Supply**\n\nCards that can never be reprinted have a price floor. Freely reprinted cards can fall when more are printed.")

    st.markdown("#### How the score is built")
    st.write("Every card is broken into six signals scored 0 to 10: play-rate, efficiency, card advantage, "
             "flexibility, format breadth, and keyword density. Roles are read straight from each card's rules text.")
    st.code("importance = 100 * ( 0.26*playRate + 0.18*efficiency + 0.14*cardAdvantage\n"
            "                   + 0.12*flexibility + 0.18*breadth + 0.12*keywords ) / 10", language="text")
    st.markdown("#### Price outlook")
    st.write("Demand drives price. The demand index is led by play-rate, then supply pressure pulls the other way.")
    st.code("demand   = 100 * ( 0.55*playRate + 0.25*flexibility + 0.20*cardAdvantage ) / 10\n"
            "drift/mo = (demand/100 - 0.50)*0.060 - (supplyRisk/10)*0.020\n"
            "proj90d  = priceNow * (1 + 3*drift/mo)", language="text")
    st.markdown("#### The machine-learning layer")
    st.write("Three families of models, all trained in-app on page load, with real validation metrics so you can see exactly how good each one is.")

    a1, a2, a3 = st.columns(3)
    with a1:
        st.markdown("**Snapshot models**")
        if ML.get("ok"):
            r2p = ML.get("price_r2", float("nan"))
            r2i = ML.get("imp_r2", float("nan"))
            st.markdown(f"Price model R2: **{r2p:.2f}**" if r2p == r2p else "Price model: training")
            st.markdown(f"Importance model R2: **{r2i:.2f}**" if r2i == r2i else "Importance: training")
            st.markdown(f"Trained on: **{ML.get('price_n', 0):,}** priced cards")
        else:
            st.caption("Not yet trained.")
    with a2:
        st.markdown("**Time-series forecaster**")
        if FORECASTER.get("ok"):
            st.markdown(f"30d forecast R2: **{FORECASTER.get('test_r2', 'n/a')}**")
            st.markdown(f"Directional accuracy: **{FORECASTER.get('dir_acc', 0)*100:.0f}%**")
            st.markdown(f"Training examples: **{FORECASTER.get('n_examples', 0):,}**")
            st.markdown(f"Cards with history: **{FORECASTER.get('n_cards', 0):,}**")
        else:
            st.caption("Trains once 30+ days of history exist. Run the daily collector to accumulate data.")
    with a3:
        st.markdown("**Similarity model**")
        st.write("Cosine nearest-neighbors in standardized feature space. Powers deck synergy and card recommendations.")
        if ML.get("ok"):
            st.markdown(f"Indexed: **{len(ML_NAMES):,}** cards")

    if ML.get("ok") and ML.get("price_feat_imp"):
        st.markdown("**What drives price: top feature importances from the price model**")
        try:
            import altair as alt
            fi = ML["price_feat_imp"]
            fdf = (pd.DataFrame({"Feature": [ML_FEATURE_LABELS.get(k, k) for k in fi],
                                  "Importance": list(fi.values())})
                   .sort_values("Importance", ascending=False).head(12))
            fbar = alt.Chart(fdf).mark_bar(color=ACCENT, cornerRadius=3).encode(
                y=alt.Y("Feature:N", sort="-x", title=None),
                x=alt.X("Importance:Q", title="Feature importance"),
                tooltip=["Feature", alt.Tooltip("Importance:Q", format=".3f")]
            ).properties(height=340)
            st.altair_chart(fbar, use_container_width=True)
        except Exception:
            pass

    st.code(
        "price model    : HistGradientBoostingRegressor  -> log(price), 5-fold CV R2\n"
        "importance mdl : GradientBoostingRegressor       -> learned power 0-100, 5-fold CV R2\n"
        "forecaster     : GradientBoostingRegressor       -> 30d log-return, time-based split\n"
        "similarity     : NearestNeighbors cosine         -> feature-space distance\n"
        "features       : 26 card attributes + 10 text signals + 9 time-series features",
        language="text"
    )
    st.write("The forecaster uses a time-based split: oldest 80% trains, newest 20% tests. "
             "This reflects real out-of-sample accuracy, not overfitting. The directional accuracy "
             "metric tells you what fraction of up/down calls were correct.")

    st.markdown("#### How the builder works")
    st.write("Give the brief and optionally seed a few cards. The builder searches across all of Magic, "
             "then the similarity model ranks candidates by a blend of learned power and how closely each "
             "card matches the deck's evolving fingerprint, the average feature vector of the cards already "
             "chosen. It fills the strategy's role targets card by card, re-scoring synergy as the deck grows, "
             "so the result coheres around your build-around cards instead of being a pile of unrelated good cards. "
             "A validation step then checks the finished deck against your brief.")
    st.caption("Prices are aggregated market values for study, not a brokerage quote, and outlooks are not "
               "financial advice. The math is kept transparent so the logic stays auditable. Data refreshes when the page loads.")
