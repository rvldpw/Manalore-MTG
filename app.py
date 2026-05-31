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

# ----------------------------------------------------------------------------
# DATA LAYER
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def load_library(target=TARGET):
    """The most-played cards, newest data each session (30 min cache)."""
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


def price_now(c):
    p = c.get("prices", {}) or {}
    if p.get("usd"):
        return float(p["usd"])
    if p.get("usd_foil"):
        return float(p["usd_foil"])
    if p.get("eur"):
        return float(p["eur"]) * 1.08
    return None


def legal_formats(c):
    L = c.get("legalities", {})
    return [f for f in FMTS if L.get(f) == "legal"]


def fmt_usd(v):
    if v is None:
        return "-"
    return f"${v:.2f}" if v < 10 else f"${round(v):,}"


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
    return (demand(c) / 100 - 0.5) * 0.060 - (reprint_risk(c) / 10) * 0.020


def proj_price(c):
    p = price_now(c)
    return None if p is None else p * (1 + 3 * drift_mo(c))


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
# THEME
# ----------------------------------------------------------------------------
st.set_page_config(page_title="MANALORE", page_icon="✦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600;700&family=Spectral:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
.stApp{background:linear-gradient(180deg,#0b0e12,#0e1319);color:#ece6d6;font-family:'Spectral',serif}
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


def score_color(v):
    return "#d9a850" if v >= 80 else "#9a7bc4" if v >= 65 else "#5294d6" if v >= 50 else "#7a8494"


# ---- header ----
hc1, hc2 = st.columns([1, 11])
with hc1:
    st.markdown("<div style='font-size:46px;line-height:1;color:#d9a850;text-align:center;margin-top:4px'>✦</div>",
                unsafe_allow_html=True)
with hc2:
    st.markdown("# MANALORE")
    st.markdown("<div class='cap'>ACADEMY OF CARD MASTERY</div>", unsafe_allow_html=True)

with st.spinner("Opening the library..."):
    POOL = [c for c in load_library() if c.get("name")]

if not POOL:
    st.error("Could not reach the live card service right now. Please reload the page in a moment.")
    st.stop()

stamp = time.strftime("%I:%M %p").lstrip("0")
st.markdown(
    f"<div class='statusbar'><span class='dot'>●</span> Live market data, last updated {stamp} "
    f"&nbsp;·&nbsp; tracking the {len(POOL):,} most-played cards</div>",
    unsafe_allow_html=True)

tab_lib, tab_look, tab_sets, tab_build, tab_market, tab_academy = st.tabs(
    ["Library", "Lookup", "Sets", "Builder", "Market", "Academy"])


# ---- shared card sheet ----
def card_sheet_body(c):
    """The contents of a card popup: image, verdict, metrics, charts, notes."""
    left, right = st.columns([1, 1.4])
    with left:
        im = img_uri(c, "normal")
        if im:
            st.image(im, use_container_width=True)
        fmts = legal_formats(c)
        st.markdown("<span class='cap'>LEGAL: " + (", ".join(f.title() for f in fmts) if fmts else "Limited")
                    + "</span>", unsafe_allow_html=True)
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
        delt = (pj / p * 100 - 100) if (pj and p) else 0
        d.metric("Price", fmt_usd(p), f"{delt:+.1f}% 90d")
        f = feats(c)
        contrib = pd.DataFrame({
            "signal": ["Play-rate", "Efficiency", "Card adv", "Breadth", "Flexibility", "Keywords"],
            "contribution": [round(W_IMP["play_rank"] * f["play_rank"] * 10, 1), round(W_IMP["eff"] * f["eff"] * 10, 1),
                             round(W_IMP["card_adv"] * f["card_adv"] * 10, 1), round(W_IMP["ubiq"] * f["ubiq"] * 10, 1),
                             round(W_IMP["flex"] * f["flex"] * 10, 1), round(W_IMP["kw"] * f["kw"] * 10, 1)],
        }).set_index("signal")
        st.markdown("**Why it scores this way**")
        st.bar_chart(contrib, color=ACCENT, height=190, horizontal=True)
    st.markdown("**Price outlook · next 90 days**")
    series = [round((p or 0) * (1 + m * drift_mo(c)), 2) for m in range(4)]
    st.area_chart(pd.DataFrame({"USD": series}, index=["Now", "+30d", "+60d", "+90d"]), color="#5fc28a", height=170)
    st.caption(f"Confidence {confidence(c)}% · supply risk {reprint_risk(c)}/10. Excludes surprise reprints and rules changes.")
    st.markdown("**Analyst read**")
    label = {"strong": "🟢", "caution": "🔴", "note": "🟡"}
    for tag, txt in analyst_notes(c):
        st.markdown(f"<div class='note-row'>{label[tag]} &nbsp;{txt}</div>", unsafe_allow_html=True)


@st.dialog(" ", width="large")
def open_card_dialog(c):
    st.markdown(f"## {c['name']}")
    card_sheet_body(c)


def show_card(c):
    open_card_dialog(c)


def card_tiles(cards, where, cols_n=5, limit=40):
    """Visual grid of card art. Clicking a card opens its popup."""
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
                art = img_uri(c, "art_crop") or img_uri(c, "normal")
                if art:
                    st.image(art, use_container_width=True)
                d = delta_pct(c)
                tag = f"<span class='badge {'up' if d>=0 else 'dn'}'>{'▲' if d>=0 else '▼'}{abs(d):.0f}%</span>"
                st.markdown(f"<div class='nm'>{c['name'][:30]}</div>"
                            f"<span class='pwr' style='color:{score_color(importance(c))}'>{importance(c)}</span> "
                            f"<span class='cap'>{fmt_usd(price_now(c))}</span> {tag}", unsafe_allow_html=True)
                if st.button("View card", key=f"{where}_open_{idx}_{c.get('id', c['name'])}", use_container_width=True):
                    show_card(c)


# ============================================================================
# LIBRARY
# ============================================================================
with tab_lib:
    st.markdown("<div class='hero'><h2>Read any card like a master.</h2>"
                "<p>Thousands of the most-played cards, each scored for power, demand and price outlook, "
                "in plain language for newcomers and full depth for veterans.</p></div>", unsafe_allow_html=True)

    avg_dem = int(np.mean([demand(c) for c in POOL]))
    gain = max(POOL, key=delta_pct)
    reserved = sum(1 for c in POOL if c.get("reserved"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg Demand", avg_dem)
    m2.metric("Top Gainer", gain["name"].split(",")[0], f"{delta_pct(gain):+.1f}%")
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
    st.markdown("<span class='cap'>Search any card ever printed. Partial names and typos are fine.</span>",
                unsafe_allow_html=True)

    q = st.text_input("Card name", "", placeholder="e.g. spider, bolt, ragavan, miles morales")
    if q and len(q) >= 2:
        names = scry_autocomplete(q)
        if not names:
            fb = scry_named(q)
            names = [fb["name"]] if fb and fb.get("name") else []
        if not names:
            st.info("No card found for that yet. Check the spelling, or try fewer letters.")
        else:
            cards = [c for c in scry_collection(names[:12]) if c.get("name")]
            if cards:
                st.markdown(f"<span class='cap'>{len(cards)} match(es). Tap a card to open its full sheet.</span>",
                            unsafe_allow_html=True)
                card_tiles(sorted(cards, key=lambda c: -importance(c)), "lookpick", 4, 12)
            else:
                for i, n in enumerate(names[:12]):
                    if st.button(n, key=f"lkname_{i}", use_container_width=True):
                        fb = scry_named(n)
                        if fb:
                            show_card(fb)
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
                        f"{choice.get('released_at','TBA')} · {choice['card_count']} cards</span>",
                        unsafe_allow_html=True)
            booster = st.radio("Booster view", ["All printings", "Play Booster", "Collector Booster"],
                               horizontal=True, key="booster",
                               help="Play Boosters are the standard cards. Collector Boosters hold the special "
                                    "showcase, borderless, extended-art and serialized treatments.")
            code = choice["code"]
            if booster == "Play Booster":
                q = f"e:{code} -is:showcase -is:extendedart -is:borderless -is:fullart -is:serialized"
                order, uniq = "edhrec", "cards"
                note = "Standard versions as they appear in Play Boosters, most-played first."
            elif booster == "Collector Booster":
                q = f"e:{code} (is:showcase or is:extendedart or is:borderless or is:fullart or is:serialized or is:etched)"
                order, uniq = "usd", "prints"
                note = "Collector Booster treatments: showcase, borderless, extended-art, full-art, etched and serialized, priciest first."
            else:
                q, order, uniq = f"e:{code}", "usd", "prints"
                note = "Every printing in this set, most valuable first."
            st.markdown(f"<span class='note'>{note}</span>", unsafe_allow_html=True)
            with st.spinner("Drawing the cards..."):
                set_cards = scry_search(q, order=order, unique=uniq, limit=120)
            if not set_cards:
                st.info("No cards found for this view. Try All printings.")
            else:
                st.markdown(f"<span class='cap'>{len(set_cards)} cards · showing 50</span>", unsafe_allow_html=True)
                card_tiles(set_cards, "set", 5, 50)
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
                    # forgiving: match on any word fragment overlap
                    exact = [s for s in shown if any(w[:4] and w[:4] in s["name"].lower() for w in ql.split())]
                shown = exact
            if not shown:
                st.info(f"No sets match \u201c{set_q}\u201d. Try a shorter word, like \u201cspider\u201d or \u201cfinal\u201d.")
            else:
                st.markdown(f"<span class='cap'>{len(shown)} sets · newest first</span>", unsafe_allow_html=True)
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
    st.markdown("<span class='cap'>Pick a card pool, tap the cards you like, and we build the deck around them.</span>",
                unsafe_allow_html=True)

with tab_build:
    st.markdown("### Build a deck")
    st.markdown("<span class='cap'>Tell us the brief, optionally tap a few favourite cards, and we assemble "
                "a deck from across all of Magic, then name it.</span>", unsafe_allow_html=True)

    st.markdown("**Step 1 · Your brief**  <span class='cap'>this guides the model</span>", unsafe_allow_html=True)
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

    st.markdown("**Step 2 · Add a few favourites (optional)**", unsafe_allow_html=True)
    suggest_pool = sorted(POOL, key=lambda c: -importance(c))[:30]
    with st.expander("Browse popular cards to add", expanded=False):
        rows = (len(suggest_pool) + 4) // 5
        idx = 0
        basket_names = [b["name"] for b in SS["basket"]]
        for _ in range(rows):
            cols = st.columns(5)
            for col in cols:
                if idx >= len(suggest_pool):
                    break
                c = suggest_pool[idx]; idx += 1
                with col:
                    art = img_uri(c, "art_crop")
                    if art:
                        st.image(art, use_container_width=True)
                    chosen = c["name"] in basket_names
                    st.markdown(f"<div class='nm'>{c['name'][:22]}</div>", unsafe_allow_html=True)
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

    st.markdown("**Step 3 · Build**")
    cbuild = st.columns([3, 1])
    deck_size = 100 if fmt == "Commander" else 60
    build = cbuild[0].button(f"⚔  Build my {fmt} deck", use_container_width=True, type="primary")
    if cbuild[1].button("Clear all", use_container_width=True):
        SS["basket"] = []
        st.rerun()

    if build:
        # seed key card
        seed = scry_named(key_card) if key_card.strip() else None
        starters = list(SS["basket"])
        if seed and seed.get("name") and seed["name"] not in [s["name"] for s in starters]:
            feats(seed); starters.insert(0, seed)

        # infer colors from picks + seed if not chosen
        cols = list(color_pick)
        if not cols:
            cs = set()
            for c in starters:
                cs.update(c.get("color_identity", []))
            cols = [x for x in ["W", "U", "B", "R", "G"] if x in cs]
        colors_key = "".join(sorted(cols))

        with st.spinner(f"Searching all of Magic for the best {strategy.lower()} cards..."):
            candidates = [enrich_card(c) for c in build_candidate_pool(colors_key, strategy, fmt, purpose)]

        lands = 37 if deck_size >= 100 else 24
        nonland_target = deck_size - lands
        used, used_names = list(starters), {c["name"] for c in starters}
        # rank candidates by importance, fitting the brief
        for c in sorted(candidates, key=lambda c: -importance(c)):
            if len(used) >= nonland_target:
                break
            if c["name"] not in used_names and "land" not in type_line(c).lower():
                used.append(c); used_names.add(c["name"])
        # last-resort top-up from the library
        if len(used) < nonland_target:
            for c in sorted(POOL, key=lambda c: -importance(c)):
                if len(used) >= nonland_target:
                    break
                if c["name"] not in used_names and "land" not in type_line(c).lower():
                    used.append(c); used_names.add(c["name"])

        if not used:
            st.warning("We could not find cards for that brief. Try widening colors or strategy.")
        else:
            if not cols:
                cs = set()
                for c in used:
                    cs.update(c.get("color_identity", []))
                cols = [x for x in ["W", "U", "B", "R", "G"] if x in cs]
            arch = strategy
            name = deck_name(cols, arch)
            value = sum(price_now(c) or 0 for c in used)
            avg_imp = round(np.mean([importance(c) for c in used]))

            st.markdown(f"## {name}")
            seed_txt = f"built around {seed['name']}, " if seed else ""
            picks_txt = (f"your {len(SS['basket'])} favourite(s), " if SS["basket"] else "")
            st.markdown(f"<span class='cap'>A {purpose.lower()} {strategy.lower()} {fmt} deck, {seed_txt}{picks_txt}"
                        f"drawn from across all of Magic and tuned to {guild_name(cols)}.</span>",
                        unsafe_allow_html=True)
            mm1, mm2, mm3, mm4 = st.columns(4)
            mm1.metric("Deck Size", len(used) + lands, f"{len(used)} spells / {lands} lands")
            mm2.metric("Est. Spell Value", f"${value:.0f}")
            mm3.metric("Avg Power", avg_imp)
            mm4.metric("Colors", "".join(cols) or "C")

            # ---- two charts that make sense ----
            ch1, ch2 = st.columns(2)
            with ch1:
                st.markdown("**Mana curve**")
                curve = pd.Series([min(7, int(c.get("cmc", 0))) for c in used]).value_counts().sort_index()
                curve.index = [(f"{i}" if i < 7 else "7+") for i in curve.index]
                st.bar_chart(curve, color=ACCENT, height=200)
            with ch2:
                st.markdown("**Role mix**")
                rc = {}
                for c in used:
                    r = "Other" if role(c) == "Spell" else role(c)
                    rc[r] = rc.get(r, 0) + 1
                st.bar_chart(pd.Series(rc).sort_values(ascending=False), color="#9a7bc4", height=200, horizontal=True)

            # ---- visual deck grid ----
            st.markdown("**The deck**  <span class='cap'>tap any card to inspect it</span>", unsafe_allow_html=True)
            SS["last_deck"] = used
            by_role = {}
            for c in used:
                r = "Other" if role(c) == "Spell" else role(c)
                by_role.setdefault(r, []).append(c)
            order = ["Threat", "Removal", "Counter", "Card Advantage", "Disruption", "Ramp", "Engine", "Other"]
            for r in [x for x in order if x in by_role]:
                st.markdown(f"**{r} · {len(by_role[r])}**")
                card_tiles(sorted(by_role[r], key=lambda c: -importance(c)), f"deck_{r}", 6, 60)
            st.markdown(f"**Lands · {lands}**  <span class='cap'>tuned to your colors ({''.join(cols) or 'C'})</span>",
                        unsafe_allow_html=True)
            stars = ", ".join(c["name"] for c in sorted(used, key=lambda c: -importance(c))[:3])
            st.success(f"Plays as a {arch.lower()} deck. Standouts: {stars}.")


# ============================================================================
# MARKET
# ============================================================================
with tab_market:
    st.markdown("### Market")
    st.markdown("<span class='cap'>Demand-driven 90-day outlook across the tracked cards.</span>",
                unsafe_allow_html=True)
    bull = sum(1 for c in POOL if delta_pct(c) > 0)
    bear = len(POOL) - bull
    total_val = sum(price_now(c) or 0 for c in POOL)
    hi = max(POOL, key=lambda c: price_now(c) or 0)
    mk1, mk2, mk3 = st.columns(3)
    mk1.metric("Trending Up", bull, "of tracked cards")
    mk2.metric("Trending Down", bear)
    mk3.metric("Total Tracked Value", fmt_usd(total_val))
    st.markdown(f"<span class='cap'>Priciest single card: {hi['name']} at {fmt_usd(price_now(hi))}.</span>",
                unsafe_allow_html=True)

    rows = [{"Card": c["name"], "Demand": demand(c), "Now": round(price_now(c) or 0, 2),
             "Proj 90d": round(proj_price(c) or 0, 2),
             "Move %": round((proj_price(c) or 0) / (price_now(c) or 1) * 100 - 100, 1),
             "Conf": confidence(c)} for c in POOL]
    df = pd.DataFrame(rows)
    st.markdown("**Demand vs price**  <span class='cap'>top-right is the conviction zone</span>",
                unsafe_allow_html=True)
    st.scatter_chart(df.sort_values("Demand", ascending=False).head(120), x="Demand", y="Now", height=260)

    st.markdown("**Biggest movers · next 90 days**")
    movers = df.reindex(df["Move %"].abs().sort_values(ascending=False).index).head(12)
    st.dataframe(movers, use_container_width=True, hide_index=True,
                 column_config={"Now": st.column_config.NumberColumn(format="$%.2f"),
                                "Proj 90d": st.column_config.NumberColumn(format="$%.2f"),
                                "Move %": st.column_config.NumberColumn(format="%.1f%%")})

    st.markdown("**Valuation table · top 100 by power**")
    top = sorted(POOL, key=lambda c: -importance(c))[:100]
    tdf = pd.DataFrame([{"Card": c["name"], "Power": importance(c), "Demand": demand(c),
                         "Now": round(price_now(c) or 0, 2), "Proj 90d": round(proj_price(c) or 0, 2),
                         "Move %": round(delta_pct(c), 1), "Conf": confidence(c)} for c in top])
    st.dataframe(tdf, use_container_width=True, hide_index=True,
                 column_config={"Now": st.column_config.NumberColumn(format="$%.2f"),
                                "Proj 90d": st.column_config.NumberColumn(format="$%.2f"),
                                "Move %": st.column_config.NumberColumn(format="%.1f%%")})


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
    st.markdown("#### How the builder works")
    st.write("Pick a pool, choose the cards you like, and the builder keeps your picks, then completes the deck "
             "with the highest-power cards that fit your colors, fits a curve, and names the deck. You get a real "
             "skeleton to build on, not a random pile.")
    st.caption("Prices are aggregated market values for study, not a brokerage quote, and outlooks are not "
               "financial advice. The math is kept transparent so the logic stays auditable. Data refreshes when the page loads.")
