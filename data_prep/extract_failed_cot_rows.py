import json
import os
import pandas as pd


FAILED_IDS_JSONL = "needs_shortening_salvaged.jsonl"
BASE_TRAIN_CSV = "train.csv"
OUTPUT_SALVAGE_CSV = "salvaged_puzzles_trial2.csv"

def main():
    if not os.path.exists(FAILED_IDS_JSONL):
        print(f"Error: Cannot find '{FAILED_IDS_JSONL}'.")
        return
    if not os.path.exists(BASE_TRAIN_CSV):
        print(f"Error: Cannot find your base dataset '{BASE_TRAIN_CSV}'.")
        return

    failed_ids = set()
    print(f"Reading target IDs from {FAILED_IDS_JSONL}...")
    with open(FAILED_IDS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                if "id" in record:
                    failed_ids.add(str(record["id"]))

    print(f"Found {len(failed_ids)} unique IDs to extract.")

    print(f"Loading {BASE_TRAIN_CSV}...")
    base_df = pd.read_csv(BASE_TRAIN_CSV)

    base_df['id'] = base_df['id'].astype(str)

    salvaged_df = base_df[base_df['id'].isin(failed_ids)]

    salvaged_df.to_csv(OUTPUT_SALVAGE_CSV, index=False, encoding="utf-8")

    print(f"Successfully extracted: {len(salvaged_df)} rows out of {len(failed_ids)} target IDs.")
    print(f"Saved to: {OUTPUT_SALVAGE_CSV}")
  

if __name__ == "__main__":
    main()