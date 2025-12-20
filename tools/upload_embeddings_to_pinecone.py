#!/usr/bin/env python3

import os
import json
import getpass
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from pinecone import Pinecone


# ------------------------------------------------------------
# Configuration defaults
# ------------------------------------------------------------

EMB_DIM = 960
UPSERT_BATCH_SIZE = 100  # matches original code
INDEX_NAME_DEFAULT = "imitune-search"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def open_embeddings_memmap(path, num_rows, emb_dim):
    """
    Open raw float32 embeddings written via np.memmap(..., shape=(N, D)).
    """
    return np.memmap(
        path,
        dtype="float32",
        mode="r",
        shape=(num_rows, emb_dim),
    )


# ------------------------------------------------------------
# Main upload logic
# ------------------------------------------------------------

def main(args):
    print("=" * 60)
    print("Uploading precomputed FreeSound embeddings to Pinecone")
    print("=" * 60)

    # --------------------------------------------------------
    # Load metadata (authoritative row count)
    # --------------------------------------------------------

    print(f"\n1. Loading metadata from {args.metadata_csv}...")
    meta = pd.read_csv(args.metadata_csv)

    if "freesound_url" not in meta.columns:
        raise ValueError("metadata CSV must contain 'freesound_url' column")

    num_rows = len(meta)
    print(f"   Rows: {num_rows:,}")

    # --------------------------------------------------------
    # Open embeddings (RAW memmap)
    # --------------------------------------------------------

    print(f"\n2. Opening embeddings memmap: {args.embeddings}")
    embeddings = open_embeddings_memmap(
        args.embeddings,
        num_rows=num_rows,
        emb_dim=args.embedding_dim,
    )

    print(f"   Embeddings shape: ({num_rows}, {args.embedding_dim})")

    # --------------------------------------------------------
    # Connect to Pinecone
    # --------------------------------------------------------

    print("\n3. Connecting to Pinecone...")
    api_key = os.getenv("PINECONE_API_KEY") or getpass.getpass("Pinecone API Key: ")
    if not api_key:
        raise ValueError("Pinecone API Key is required")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(args.index_name)

    print(f"   Connected to index: {args.index_name}")
    print(index.describe_index_stats())

    # --------------------------------------------------------
    # Upload loop (ID scheme EXACTLY matches original code)
    # --------------------------------------------------------

    print(f"\n4. Uploading vectors in batches of {UPSERT_BATCH_SIZE}...")

    for start in tqdm(range(0, num_rows, UPSERT_BATCH_SIZE), desc="Uploading"):
        end = min(start + UPSERT_BATCH_SIZE, num_rows)

        vectors = []
        for i in range(start, end):
            # IMPORTANT: ID scheme from original code
            # idx = stats["successful"] + 1
            # id = f"{idx:012d}"
            vector_id = f"{i + 1:012d}"

            vectors.append({
                "id": vector_id,
                "values": embeddings[i].tolist(),
                "metadata": {
                    "freesound_url": meta.iloc[i]["freesound_url"]
                },
            })

        try:
            index.upsert(vectors=vectors)
        except Exception as e:
            print(f"\n❌ Error upserting batch starting at {start}: {e}")
            raise

    print("\n✅ Upload complete.")
    print(index.describe_index_stats())


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload precomputed FreeSound embeddings to Pinecone"
    )

    parser.add_argument(
        "--embeddings",
        required=True,
        help="Path to raw float32 embeddings file (created via np.memmap)",
    )
    parser.add_argument(
        "--metadata_csv",
        required=True,
        help="CSV containing freesound_url column (row order must match embeddings)",
    )
    parser.add_argument(
        "--index_name",
        default=INDEX_NAME_DEFAULT,
        help="Pinecone index name",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=EMB_DIM,
        help="Embedding dimension (default: 960)",
    )

    args = parser.parse_args()
    main(args)
