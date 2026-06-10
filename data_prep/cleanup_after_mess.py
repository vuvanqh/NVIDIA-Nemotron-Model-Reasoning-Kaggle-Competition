import json
import os

TARGET_FILE = "flash_lite_raw_reasoning.jsonl"
TEMP_FILE = "flash_lite_clean_backup.jsonl"

def clear_duplicates():
    if not os.path.exists(TARGET_FILE):
        print(f"File {TARGET_FILE} not found.")
        return

    seen_ids = set()
    unique_records = []
    duplicate_count = 0

    print(f"Reading records from {TARGET_FILE}...")
    with open(TARGET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                r_id = str(data["id"])
                
                # Check by ID if the puzzle was already solved or not
                if r_id not in seen_ids:
                    seen_ids.add(r_id)
                    unique_records.append(data)
                else:
                    duplicate_count += 1
            except Exception as e:
                continue

    print(f"Found {len(unique_records)} unique puzzles and dropped {duplicate_count} duplicates.")


    print(f"Saving pristine rows to backup...")
    with open(TEMP_FILE, "w", encoding="utf-8") as f:
        for record in unique_records:
            f.write(json.dumps(record) + "\n")

    os.replace(TEMP_FILE, TARGET_FILE)
    print(f"Success! {TARGET_FILE} is now completely unique and deduplicated.")

if __name__ == "__main__":
    clear_duplicates()