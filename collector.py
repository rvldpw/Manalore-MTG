"""
MANALORE data collector
========================
Snapshots full Magic card data with prices to a Hugging Face dataset, once per run.
Designed to run on a daily schedule (see .github/workflows/collect.yml).

What it does:
  1. Downloads Scryfall's "default_cards" bulk file (every paper printing, current prices).
  2. Keeps the full record per card but stores it efficiently as Parquet.
  3. Writes one dated partition: data/snapshot_date=YYYY-MM-DD/cards.parquet
  4. Pushes that partition to the Hugging Face dataset repo.

Over time this builds full price history across all cards and all periods, which is
exactly what a real time-series price model needs.

Environment variables required:
  HF_TOKEN        a Hugging Face token with WRITE access to the dataset repo
  HF_DATASET_REPO e.g. "your-username/manalore-mtg-history"

Run locally:
  pip install requests pandas pyarrow huggingface_hub ijson
  HF_TOKEN=... HF_DATASET_REPO=you/manalore-mtg-history python collector.py

Notes:
  - The full snapshot is large (~450 MB JSON). We stream-parse it so memory stays low,
    keep the fields that matter for analysis, and write compressed Parquet (~tens of MB).
  - "Full card data" here means every printing and every analysis-relevant field; we drop
    only heavy, non-analytical blobs (rulings URIs, related-card URIs, large image dicts)
    while keeping image_uris.normal/art_crop so the app can still render cards.
"""

import os
import io
import sys
import json
import time
import datetime as dt

import requests

SCRYFALL_BULK = "https://api.scryfall.com/bulk-data"
HEADERS = {"User-Agent": "Manalore-Collector/1.0", "Accept": "application/json"}

# Fields we keep per card. Everything needed for pricing, scoring, ML, and rendering.
KEEP_FIELDS = [
    "id", "oracle_id", "name", "released_at", "set", "set_name", "set_type",
    "collector_number", "rarity", "mana_cost", "cmc", "type_line", "oracle_text",
    "colors", "color_identity", "keywords", "produced_mana", "power", "toughness",
    "loyalty", "legalities", "reserved", "foil", "nonfoil", "finishes",
    "promo", "promo_types", "frame_effects", "border_color", "full_art",
    "prices", "edhrec_rank", "penny_rank", "digital", "layout",
]


def get_bulk_download_url():
    r = requests.get(SCRYFALL_BULK, headers=HEADERS, timeout=30)
    r.raise_for_status()
    for entry in r.json().get("data", []):
        if entry.get("type") == "default_cards":
            return entry["download_uri"]
    raise RuntimeError("Could not find default_cards bulk entry")


def slim(card):
    """Keep only analysis-relevant fields; trim image dict to two sizes."""
    out = {k: card.get(k) for k in KEEP_FIELDS if k in card}
    iu = card.get("image_uris") or {}
    out["img_normal"] = iu.get("normal")
    out["img_art_crop"] = iu.get("art_crop")
    # double-faced cards keep front-face essentials so the app can still render/score them
    if not out.get("oracle_text") and card.get("card_faces"):
        f0 = card["card_faces"][0]
        out["oracle_text"] = f0.get("oracle_text")
        out["type_line"] = out.get("type_line") or f0.get("type_line")
        fiu = f0.get("image_uris") or {}
        out["img_normal"] = out["img_normal"] or fiu.get("normal")
        out["img_art_crop"] = out["img_art_crop"] or fiu.get("art_crop")
    return out


def stream_cards(url):
    """Stream-parse the big JSON array so we never hold it all in memory at once."""
    try:
        import ijson
    except ImportError:
        # fallback: download and json.loads (uses more memory but works)
        data = requests.get(url, headers=HEADERS, timeout=600).json()
        for c in data:
            yield c
        return
    with requests.get(url, headers=HEADERS, timeout=600, stream=True) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        for card in ijson.items(r.raw, "item"):
            yield card


def main():
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_DATASET_REPO")
    if not token or not repo:
        print("ERROR: set HF_TOKEN and HF_DATASET_REPO environment variables.", file=sys.stderr)
        sys.exit(1)

    import pandas as pd

    today = dt.date.today().isoformat()
    print(f"[{today}] Fetching Scryfall bulk URL...")
    url = get_bulk_download_url()
    print(f"Streaming cards from {url}")

    rows, kept = [], 0
    for card in stream_cards(url):
        if card.get("digital"):
            continue  # paper only
        row = slim(card)
        row["snapshot_date"] = today
        # store nested dicts (prices, legalities) as JSON strings for clean Parquet
        for nested in ("prices", "legalities"):
            if isinstance(row.get(nested), (dict, list)):
                row[nested] = json.dumps(row[nested])
        for listf in ("colors", "color_identity", "keywords", "produced_mana",
                      "finishes", "promo_types", "frame_effects"):
            if isinstance(row.get(listf), list):
                row[listf] = json.dumps(row[listf])
        rows.append(row)
        kept += 1
        if kept % 20000 == 0:
            print(f"  ...{kept} cards")

    print(f"Kept {kept} paper cards. Building Parquet...")
    df = pd.DataFrame(rows)

    # write Parquet to a temp file
    local_dir = "hf_upload"
    os.makedirs(local_dir, exist_ok=True)
    out_path = os.path.join(local_dir, "cards.parquet")
    df.to_parquet(out_path, compression="zstd", index=False)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path} ({size_mb:.1f} MB)")

    # upload to HF under a dated partition
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
    path_in_repo = f"data/snapshot_date={today}/cards.parquet"
    print(f"Uploading to {repo}:{path_in_repo} ...")
    api.upload_file(path_or_fileobj=out_path, path_in_repo=path_in_repo,
                    repo_id=repo, repo_type="dataset",
                    commit_message=f"Snapshot {today} ({kept} cards)")
    print("Done.")


if __name__ == "__main__":
    main()
