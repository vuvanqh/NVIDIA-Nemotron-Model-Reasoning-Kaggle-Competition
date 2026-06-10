import json
import os
import pandas as pd
from transformers import AutoTokenizer

INPUT_FILE = "needs_gemma_shortening.jsonl"

tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct", use_fast=False)

def inspect_red_cohort():
    if not os.path.exists(INPUT_FILE):
        print(f"Cannot find '{INPUT_FILE}'.")
        return

    lengths = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                lengths.append(record["nemotron_tokens"])

    df = pd.Series(lengths)
    
    just_barely = sum(1 for x in lengths if 1025 <= x <= 1150)
    moderate = sum(1 for x in lengths if 1151 <= x <= 1300)
    massive = sum(1 for x in lengths if x > 1300)


    print(f"Total Rows to Fix:        {len(df)}")
    print(f"Average Token Length:     {int(df.mean())} tokens")
    print(f"Min Length in this Pack:  {df.min()} tokens")
    print(f"Max Length in this Pack:  {df.max()} tokens")

    print(f"Just Barely Over (1025-1150): {just_barely} ({just_barely/len(df)*100:.1f}%)")
    print(f"Moderately Over (1151-1300):  {moderate} ({moderate/len(df)*100:.1f}%)")
    print(f"Massively Over (>1300):       {massive} ({massive/len(df)*100:.1f}%)")

if __name__ == "__main__":
    inspect_red_cohort()