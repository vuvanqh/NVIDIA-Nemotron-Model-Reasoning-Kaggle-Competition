import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

READY_TO_TRAIN = "nemotron_ready_to_train.jsonl"
TRAIN_OUTPUT = "nemotron_train_v1.jsonl"
VAL_OUTPUT = "nemotron_val_v1.jsonl"

def classify_task(prompt):
    prompt_lower = str(prompt).lower()
    if "bit manipulation" in prompt_lower:
        return "Bit Manipulation"
    elif "encryption rules" in prompt_lower:
        return "Text Encryption"
    elif "numeral system" in prompt_lower:
        return "Numeral Conversion"
    elif "unit conversion" in prompt_lower:
        return "Unit Conversion"
    elif "applied to equations" in prompt_lower:
        return "Equations & Symbolic"
    elif "gravitational constant" in prompt_lower:
        return "Physics Gravity"
    else:
        return "Unknown"

def main():
    if not os.path.exists(READY_TO_TRAIN):
        print(f"Cannot find '{READY_TO_TRAIN}'.")
        return

    pool = []
    
    with open(READY_TO_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
            
                record = json.loads(line)
                puzzle_raw = record["puzzle"].strip()
                
                task_type = classify_task(puzzle_raw)
                
                payload = {
                    "id": record["id"],
                    "task_type": task_type,
                    "puzzle": puzzle_raw, 
                    "target_answer": record.get("target_answer", ""),
                    "output": record["output"].strip(),
                    "nemotron_tokens": record.get("nemotron_tokens", None)
                }
                pool.append(payload)

    df_final = pd.DataFrame(pool)

    train_df, val_df = train_test_split(
        df_final, test_size=0.15, stratify=df_final['task_type'], random_state=42
    )

    for dataset, filename in [(train_df, TRAIN_OUTPUT), (val_df, VAL_OUTPUT)]:
        with open(filename, "w", encoding="utf-8") as outfile:
            for _, row in dataset.iterrows():
                outfile.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")

    print("\n" + "="*50)
    print("DATASET READY")
    print("="*50)
    print(f"Training Set Size:  {len(train_df)}")
    print(f"Validation Set Size: {len(val_df)}")
    print(f"Safe outputs written to: {TRAIN_OUTPUT} and {VAL_OUTPUT}")
    print("="*50)

if __name__ == "__main__":
    main()

