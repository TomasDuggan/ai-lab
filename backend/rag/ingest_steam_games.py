"""
Ingest Steam game data from appids and store in JSONL format.

Reads appids from appids.txt, fetches data from Steam API,
cleans and extracts relevant fields, saves to backend/data/raw/games.jsonl.
"""

import json
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any
import urllib.request
import urllib.error


"""Load and deduplicate appids from text file (comma-separated)."""
def load_appids(file_path: Path) -> set[int]:
    with open(file_path, 'r') as f:
        content = f.read()

    # Parse comma-separated values across multiple lines
    ids_str = content.replace('\n', ',')
    ids_list = [int(id.strip()) for id in ids_str.split(',') if id.strip()]

    deduplicated = set(ids_list)
    print(f"Loaded {len(ids_list)} appids, {len(deduplicated)} unique after dedup")
    return deduplicated


"""Clean HTML tags and normalize whitespace."""
def clean_html(html_text: str) -> str:
    if not html_text:
        return ""

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', html_text)

    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&quot;', '"')
    text = text.replace('&apos;', "'")
    text = text.replace('&amp;', '&')
    text = text.replace('<br>', ' ')
    text = text.replace('<br/>', ' ')
    text = text.replace('<br />', ' ')

    # Normalize whitespace: collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


"""Fetch app details from Steam API."""
def fetch_app_details(appid: int) -> Optional[Dict[str, Any]]:
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}&l=english"

    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

        if str(appid) not in data:
            return None

        app_data = data[str(appid)]

        # Check if request was successful
        if not app_data.get('success', False):
            return None

        return app_data.get('data', {})

    except (urllib.error.URLError, json.JSONDecodeError, Exception) as e:
        print(f"  ERROR fetching {appid}: {type(e).__name__}")
        return None


"""Extract and clean relevant fields from app data."""
def extract_fields(appid: int, app_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        extracted = {
            'appid': appid,
            'name': app_data.get('name', ''),
            'short_description': app_data.get('short_description', ''),
            'about_the_game': clean_html(app_data.get('about_the_game', '')),
            'release_date': app_data.get('release_date', {}).get('date', ''),
        }

        # Extract genres (list of objects with 'id' and 'description')
        genres = app_data.get('genres', [])
        extracted['genres'] = [g.get('description', '') for g in genres if isinstance(g, dict)]

        # Extract developers
        developers = app_data.get('developers', [])
        extracted['developers'] = developers if isinstance(developers, list) else []

        # Extract publishers
        publishers = app_data.get('publishers', [])
        extracted['publishers'] = publishers if isinstance(publishers, list) else []

        # Extract categories (list of objects with 'id' and 'description')
        categories = app_data.get('categories', [])
        extracted['categories'] = [c.get('description', '') for c in categories if isinstance(c, dict)]

        # Only return if we got the core fields
        if extracted['name']:
            return extracted

    except Exception as e:
        print(f"  ERROR extracting fields for {appid}: {e}")

    return None


"""Main ingestion pipeline."""
def main():
    appids_file = Path(__file__).parent / 'data' / 'raw' / 'steam_appids.txt'
    output_file = Path(__file__).parent / 'data' / 'raw' / 'games.jsonl'

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load appids
    appids = load_appids(appids_file)
    appids_list = sorted(list(appids))

    print(f"\nStarting ingestion of {len(appids_list)} games...")
    print("-" * 60)

    successful = 0
    failed = 0
    failed_ids = []

    # Open output file for streaming writes
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, appid in enumerate(appids_list, 1):
            # Fetch data
            app_data = fetch_app_details(appid)

            if app_data is None:
                failed += 1
                failed_ids.append(appid)
                print(f"[{idx}/{len(appids_list)}] FAIL: {appid}")
            else:
                # Extract fields
                extracted = extract_fields(appid, app_data)

                if extracted:
                    # Write to JSONL
                    out_f.write(json.dumps(extracted, ensure_ascii=False) + '\n')
                    game_name = extracted.get('name', 'Unknown')
                    print(f"[{idx}/{len(appids_list)}] OK: {game_name}")
                    successful += 1
                else:
                    failed += 1
                    failed_ids.append(appid)
                    print(f"[{idx}/{len(appids_list)}] FAIL (extraction): {appid}")

            # Rate limit: 1 request per second
            time.sleep(1)

    # Summary
    print("-" * 60)
    print(f"\nIngestion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if failed_ids:
        print(f"Failed appids: {failed_ids}")
    print(f"Output: {output_file}")


if __name__ == '__main__':
    main()
