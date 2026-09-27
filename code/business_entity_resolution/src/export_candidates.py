"""Export the full blocking candidates into candidate_pairs.tsv."""
import sys
from pathlib import Path
import polars as pl

from config import cache_path, OUTPUT_DIR
from predict import write_grouped

def main():
    print("Reading test entity ID tables...")
    s1_norm = pl.read_parquet(cache_path("test_source1_norm.parquet"), columns=["entity_id"])
    s1_ids = s1_norm["entity_id"]
    
    q_s2 = pl.read_parquet(cache_path("test_source2_norm.parquet"), columns=["entity_id"])
    q_s3 = pl.read_parquet(cache_path("test_source3_norm.parquet"), columns=["entity_id"])
    q_ids = pl.concat([q_s2["entity_id"], q_s3["entity_id"]])
    
    cand_dir = cache_path("v4test_cands")
    if not cand_dir.exists() or not list(cand_dir.glob("*.parquet")):
        cand_dir = cache_path("test_cands")
        
    print(f"Reading candidate parquet chunks from {cand_dir}...")
    chunk_files = sorted(cand_dir.glob("*.parquet"))
    if not chunk_files:
        print("ERROR: No candidate parquet files found! Run candidates.py first.")
        sys.exit(1)
        
    chunks = []
    for f in chunk_files:
        chunks.append(pl.read_parquet(f, columns=["s1_idx", "q_idx"], memory_map=False))
        
    cand = pl.concat(chunks)
    print(f"Total candidate pairs: {cand.height:,}")
    
    out_file = OUTPUT_DIR / "candidate_pairs.tsv"
    print(f"Writing candidate_pairs.tsv to {out_file}...")
    n, n_c = write_grouped(s1_ids, q_ids, cand, "candidate_entity_ids", out_file)
    print(f"DONE! Wrote {n:,} S1 rows ({n_c:,} with candidates) to {out_file}")

if __name__ == "__main__":
    main()
