import json
import os


# The JSONL file containing the IDs to filter BY
#FILTER_IDS_JSONL = "needs_shortening_salvaged.jsonl" 
FILTER_IDS_JSONL = "needs_shortening_salvaged_trial2.jsonl" 

SOURCE_DATA_JSONL = "salvaged_flash_lite_raw_reasoning_trial2.jsonl"  


OUTPUT_SALVAGE_JSONL = "manunal.jsonl"

def main():
    if not os.path.exists(FILTER_IDS_JSONL):
        print(f"Error: Cannot find your filter file '{FILTER_IDS_JSONL}'.")
        return
    if not os.path.exists(SOURCE_DATA_JSONL):
        print(f"Error: Cannot find your source dataset '{SOURCE_DATA_JSONL}'.")
        return

    
    failed_ids = set()
    print(f"Reading target IDs from {FILTER_IDS_JSONL}...")
    
    with open(FILTER_IDS_JSONL, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if "id" in record:
                    failed_ids.add(str(record["id"]))
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {line_num} of filter file.")

    print(f"Found {len(failed_ids)} unique target IDs to extract.")

    print(f"Streaming and filtering {SOURCE_DATA_JSONL}...")
    matched_count = 0
    
    with open(SOURCE_DATA_JSONL, "r", encoding="utf-8") as infile, \
         open(OUTPUT_SALVAGE_JSONL, "w", encoding="utf-8") as outfile:
        
        for line_num, line in enumerate(infile, start=1):
            if not line.strip():
                continue  # Skip blank lines
                
            try:
                record = json.loads(line)
                if "id" in record and str(record["id"]) in failed_ids:
                    outfile.write(json.dumps(record) + "\n")
                    matched_count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON on line {line_num} of source file.")

    
    print(f"Successfully extracted: {matched_count} rows out of {len(failed_ids)} target IDs.")
    print(f"Saved to: {OUTPUT_SALVAGE_JSONL}")
    

if __name__ == "__main__":
    main()