import os
import json
import pandas as pd
import argparse

def load_metadata_file(path):
    with open(path, "r") as f:
        return json.load(f)

def get_freesound_id(filepath):
    return os.path.splitext(os.path.basename(filepath))[0]

def find_metadata_file(filepath):
    if "dev_audio" in filepath:
        return "dev_clips_info_FSD50K.json"
    elif "eval_audio" in filepath:
        return "eval_clips_info_FSD50K.json"
    else:
        raise ValueError(f"Can't determine metadata file for path: {filepath}")

def construct_url(fsid, uploader):
    return f"https://freesound.org/people/{uploader}/sounds/{fsid}/"

def main():
    parser = argparse.ArgumentParser(description="Add Freesound URLs to FSD50K file list.")
    parser.add_argument("--files_csv", required=True, help="CSV with column of audio file paths")
    parser.add_argument("--filename_col", default="filename", help="Column with relative or absolute paths to audio files")
    parser.add_argument("--metadata_dir", required=True, help="Directory containing dev/eval metadata files")
    parser.add_argument("--out_csv", required=True, help="Where to save the updated CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.files_csv)
    if args.filename_col not in df.columns:
        raise ValueError(f"'{args.filename_col}' not found in columns: {df.columns.tolist()}")

    # Cache metadata files after first load
    metadata_cache = {}

    urls = []
    for path in df[args.filename_col]:
        fsid = get_freesound_id(path)
        metadata_file = find_metadata_file(path)
        metadata_path = os.path.join(args.metadata_dir, metadata_file)

        if metadata_file not in metadata_cache:
            print(f"Loading metadata: {metadata_file}")
            metadata_cache[metadata_file] = load_metadata_file(metadata_path)

        meta = metadata_cache[metadata_file]
        if fsid not in meta:
            print(f"Warning: ID {fsid} not found in {metadata_file}")
            urls.append(None)
            continue

        uploader = meta[fsid]["uploader"]
        url = construct_url(fsid, uploader)
        urls.append(url)

    df["freesound_url"] = urls
    df.to_csv(args.out_csv, index=False)
    print(f"Saved updated CSV with Freesound URLs to {args.out_csv}")

if __name__ == "__main__":
    main()
