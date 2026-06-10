import json
from collections import Counter

INPUT_JSONL = "whole_data_with_cot_to_process.jsonl"  

def check_duplicate_ids(filename):
    ids = []
    total_rows = 0
    corrupt_rows = 0

    print(f"Reading {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            total_rows += 1
            try:
                data = json.loads(line)
                ids.append(str(data["id"]))
            except KeyError:
                print(f"Warning: Line {line_num} is missing an 'id' field.")
                corrupt_rows += 1
            except json.JSONDecodeError:
                print(f"Warning: Line {line_num} is not valid JSON.")
                corrupt_rows += 1


    id_counts = Counter(ids)
    duplicates = {id_: count for id_, count in id_counts.items() if count > 1}
    unique_count = len(id_counts)

    print("\n" + "="*40)
    print("📋 PRE-FLIGHT ID INTEGRITY REPORT")
    print("="*40)
    print(f"Total rows scanned:     {total_rows}")
    print(f"Unique puzzle IDs:      {unique_count}")
    print(f"Corrupt/Invalid rows:   {corrupt_rows}")
    
    if duplicates:
        print(f"STATUS: DUPLICATES DETECTED! ({len(duplicates)} duplicate IDs found)")
        print("\nBreakdown of duplicate IDs (ID: Occurrences):")
        for dup_id, count in list(duplicates.items())[:10]: 
            print(f"  - {dup_id}: {count} times")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more duplicate IDs.")
    else:
        print("STATUS: PERFECT DATA INTEGRITY. All IDs are 100% unique.")
    print("="*40)

if __name__ == "__main__":
    check_duplicate_ids(INPUT_JSONL)