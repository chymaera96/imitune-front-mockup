import argparse
import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import random
import json

# ------------------------------------------------------------
# URL check logic
# ------------------------------------------------------------

def is_valid_url(url, timeout=5):
    """
    Returns True if URL responds with status < 400.
    Uses HEAD to avoid downloading content.
    """
    try:
        r = requests.head(
            url,
            allow_redirects=True,
            timeout=timeout,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; dataset-validation/1.0)"
            },
        )
        return r.status_code < 400
    except requests.RequestException:
        return False


# ------------------------------------------------------------
# Worker wrapper (with retry)
# ------------------------------------------------------------

def check_with_retry(url, retries=2, sleep_sec=1):
    for attempt in range(retries + 1):
        if is_valid_url(url):
            return True
        sleep_sec = 2 ** attempt
        time.sleep(sleep_sec)
    return False


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main(args):
    df = pd.read_csv(args.input_csv)

    # Preserve original embedding indices
    df["embedding_index"] = df.index

    # Drop missing URLs (these are automatically invalid)
    df = df.dropna(subset=["freesound_url"])
    df = df[df["freesound_url"].str.strip() != ""]
    
    # Reset index after filtering
    df = df.reset_index(drop=True)

    urls = df["freesound_url"].tolist()

    print(f"Checking {len(urls)} URLs...")

    valid_mask = [False] * len(urls)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(check_with_retry, url): i
            for i, url in enumerate(urls)
        }

        for idx, future in enumerate(as_completed(futures)):
            i = futures[future]
            try:
                valid_mask[i] = future.result()
            except Exception as e:
                valid_mask[i] = False
                print(f"Error checking URL at index {i}: {e}")

            if (idx + 1) % args.log_every == 0:
                print(f"Checked {idx + 1}/{len(urls)} URLs")

    # Split valid / invalid using boolean indexing
    df_valid = df[valid_mask].copy()
    df_invalid = df[[not v for v in valid_mask]].copy()

    # Save valid rows
    df_valid[["filepath", "freesound_url"]].to_csv(
        args.output_csv,
        index=False,
    )

    # Save deleted embedding indices as JSON
    deleted_indices_path = args.output_csv.replace(".csv", "_deleted_indices.json")

    deleted_indices = df_invalid["embedding_index"].tolist()

    with open(deleted_indices_path, "w") as f:
        json.dump(deleted_indices, f)

    print("Done.")
    print(f"Valid URLs: {len(df_valid)} / {len(df)}")
    print(f"Saved valid URLs to: {args.output_csv}")
    print(f"Saved deleted embedding indices to: {deleted_indices_path}")



# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)

    parser.add_argument("--workers", type=int, default=5,
                        help="Concurrent requests (keep low to avoid blocking)")
    parser.add_argument("--sleep", type=float, default=0.1,
                        help="Sleep between requests (seconds)")
    parser.add_argument("--log_every", type=int, default=1000)

    args = parser.parse_args()

    main(args)
