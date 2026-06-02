"""
MANALORE historical price importer (one-time seed)
==================================================
Seeds your Hugging Face dataset with PUBLIC historical MTG prices so the
forecasting model has a running start instead of waiting weeks for the daily
collector to accumulate data.

Source: MTGJSON (https://mtgjson.com), which publishes free, public price
history. We use two of its files:
  - AllPricesToday.json  (or AllPrices.json for deeper history if you prefer)
  - AllPrintings or the lighter "card identifiers" to map UUID -> card name/set.

What it writes:
  Exactly the same layout your collector + app already read:
  data/snapshot_date=YYYY-MM-DD/cards.parquet, one partition per historical day,
  with at least: snapshot_date, name, set, prices (JSON with "usd").
So once this runs, the app's history charts and (later) the forecaster light up
with months of real data.

Environment variables:
  HF_TOKEN         HF token with WRITE access to the dataset repo
  HF_DATASET_REPO  e.g. "you/manalore-mtg-history"

Optional:
  MTGJSON_PRICES_URL  override the prices file (default: AllPricesToday.json.gz)
  MAX_DAYS            cap how many historical days to import (default: all available)

Run once:
  pip install requests pandas pyarrow huggingface_hub ijson
  HF_TOKEN=... HF_DATASET_REPO=you/manalore-mtg-history python import_history.py

Honest notes:
  - MTGJSON price history typically covers the trailing ~3 months of daily points,
    not multiple years. That is the best free public source; the clock for deeper
    history then continues via your daily collector.
  - We import the paper "retail normal" USD price per card per day. Foil and other
    providers exist in the file; we keep it to one clean USD series to match the app.
  - This writes one partition per day. Days that already exist in the repo are
    skipped, so it is safe to re-run.
"""

import os
import sys
import gzip
import json
import io
import datetime as dt

import requests

# MTGJSON files. AllPricesToday is smaller (recent days); AllPrices.json.gz is the
# full history file. Default to the fuller history file for a better seed.
PRICES_URL = os.environ.get("MTGJSON_PRICES_URL", "https://mtgjson.com/api/v5/AllPrices.json.gz")
IDENT_URL = "https://mtgjson.com/api/v5/AllIdentifiers.json.gz"
HEADERS = {"User-Agent": "Manalore-Importer/1.0"}


def download_gz_json(url):
    """Download a .gz JSON file and return the parsed object (streamed decompress)."""
    print(f"Downloading {url} ...")
    r = requests.get(url, headers=HEADERS, timeout=900, stream=True)
    r.raise_for_status()
    buf = io.BytesIO(r.content)
    with gzip.GzipFile(fileobj=buf) as gz:
        return json.load(gz)


def build_uuid_map():
    """Map MTGJSON UUID -> (name, set code) using AllIdentifiers."""
    print("Building UUID -> card map from AllIdentifiers...")
    data = download_gz_json(IDENT_URL)
    cards = data.get("data", {})
    out = {}
    for uuid, card in cards.items():
        out[uuid] = (card.get("name"), (card.get("setCode") or "").lower())
    print(f"  mapped {len(out):,} card UUIDs")
    return out


def extract_usd_series(price_block):
    """From an MTGJSON per-card price block, return {date: usd_float} for paper retail normal.
    Structure: price_block['paper'][provider]['retail']['normal'][date] = price."""
    series = {}
    paper = (price_block or {}).get("paper", {})
    for provider, pdata in paper.items():
        retail = (pdata or {}).get("retail", {})
        normal = (retail or {}).get("normal", {})
        for date, price in normal.items():
            # if multiple providers, keep the first seen / average lightly
            try:
                v = float(price)
            except (TypeError, ValueError):
                continue
            if date in series:
                series[date] = round((series[date] + v) / 2, 2)
            else:
                series[date] = round(v, 2)
    return series


def main():
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_DATASET_REPO")
    if not token or not repo:
        print("ERROR: set HF_TOKEN and HF_DATASET_REPO.", file=sys.stderr)
        sys.exit(1)
    max_days = int(os.environ.get("MAX_DAYS", "0")) or None

    import pandas as pd
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)

    # which days already exist, so re-runs skip them
    existing = set()
    try:
        for f in api.list_repo_files(repo_id=repo, repo_type="dataset"):
            if f.startswith("data/prices/snapshot_date=") and f.endswith(".parquet"):
                existing.add(f.split("snapshot_date=")[1].split("/")[0])
    except Exception:
        pass
    print(f"{len(existing)} day-partitions already in the repo (will skip those).")

    uuid_map = build_uuid_map()

    prices = download_gz_json(PRICES_URL)
    pdata = prices.get("data", {})
    print(f"Parsing prices for {len(pdata):,} cards into per-day tables...")

    # accumulate: date -> list of {name, set, prices}
    by_day = {}
    for uuid, block in pdata.items():
        name, setcode = uuid_map.get(uuid, (None, None))
        if not name:
            continue
        series = extract_usd_series(block)
        for date, usd in series.items():
            by_day.setdefault(date, []).append(
                {"snapshot_date": date, "name": name, "set": setcode,
                 "prices": json.dumps({"usd": f"{usd}"})})

    all_days = sorted(by_day.keys())
    if max_days:
        all_days = all_days[-max_days:]
    print(f"Found {len(all_days)} historical days. Uploading new ones...")

    os.makedirs("hf_import", exist_ok=True)
    uploaded = 0
    for date in all_days:
        if date in existing:
            continue
        rows = by_day[date]
        # collapse duplicate (name,set) rows to one mean price per card per day
        df = pd.DataFrame(rows)
        df["usd_tmp"] = df["prices"].map(lambda s: float(json.loads(s)["usd"]))
        df = (df.groupby(["snapshot_date", "name", "set"], as_index=False)["usd_tmp"].mean())
        df["prices"] = df["usd_tmp"].map(lambda v: json.dumps({"usd": f"{round(v,2)}"}))
        df = df.drop(columns=["usd_tmp"])

        out_path = os.path.join("hf_import", "cards.parquet")
        df.to_parquet(out_path, compression="zstd", index=False)
        api.upload_file(path_or_fileobj=out_path,
                        path_in_repo=f"data/prices/snapshot_date={date}/cards.parquet",
                        repo_id=repo, repo_type="dataset",
                        commit_message=f"Historical seed {date} ({len(df)} cards)")
        uploaded += 1
        if uploaded % 10 == 0:
            print(f"  uploaded {uploaded} day-partitions...")

    print(f"Done. Uploaded {uploaded} new historical day-partitions to {repo}.")
    print("Your app will now show real history; once ~30+ days exist, the forecaster can train.")


if __name__ == "__main__":
    main()
