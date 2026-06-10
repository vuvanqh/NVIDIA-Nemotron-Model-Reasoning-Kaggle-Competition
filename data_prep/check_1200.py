import json
import os

INPUT_FILE = "needs_gemma_shortening.jsonl"

def count_truncated_rows():
    if not os.path.exists(INPUT_FILE):
        print(f"Cannot find '{INPUT_FILE}'.")
        return

    total_oversized = 0
    hard_cutoff_count = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                total_oversized += 1
                
                raw_trace = record.get("raw_reasoning_to_shorten", record.get("output", "")).strip()
                if not raw_trace.endswith(('.', '}', '**', 'm', 's', '"', '`')):
                    hard_cutoff_count += 1

    print("\n" + "="*45)
    print("CUT-OFF DIAGNOSTIC REPORT")
    print("="*45)
    print(f"Total Rows in this pack:     {total_oversized}")
    print(f"Estimated Hard Cut-offs:     {hard_cutoff_count} ({hard_cutoff_count/total_oversized*100:.1f}%)")
    print("="*45)

if __name__ == "__main__":
    count_truncated_rows()