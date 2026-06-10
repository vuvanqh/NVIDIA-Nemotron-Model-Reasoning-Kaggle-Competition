import json
import os
import pandas as pd

READY_TO_TRAIN = "nemotron_ready_to_train.jsonl"
DROPPED_FILE = "needs_gemma_shortening.jsonl"

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

def analyze_category_skew():
    counts = {}

    def parse_file(filepath, bucket_name):
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        puzzle_text = record.get("puzzle", record.get("input", ""))
                        
                        cat = classify_task(puzzle_text)
                        
                        if cat not in counts:
                            counts[cat] = {"ready": 0, "dropped": 0}
                        counts[cat][bucket_name] += 1
                    except Exception as e:
                        continue

    print("Scanning file matrices to trace category balance profiles...")
    parse_file(READY_TO_TRAIN, "ready")
    parse_file(DROPPED_FILE, "dropped")

    if not counts:
        print("No valid samples found across file arrays.")
        return

    report_data = []
    for cat, data in counts.items():
        total = data["ready"] + data["dropped"]
        drop_pct = (data["dropped"] / total) * 100 if total > 0 else 0
        report_data.append({
            "Category": cat,
            "Retained (V1)": data["ready"],
            "Dropped (Tomorrow)": data["dropped"],
            "Total Matrix": total,
            "Drop Rate": f"{drop_pct:.1f}%"
        })

    df_report = pd.DataFrame(report_data).sort_values(by="🟢 Retained (V1)", ascending=False)

    print("\n" + "="*75)
    print("TASK CATEGORY PROFILE DATA SKEW DIAGNOSTIC")
    print("="*75)
    print(df_report.to_string(index=False))
    print("="*75)
    
    completely_lost = [row["Category"] for row in report_data if row["🟢 Retained (V1)"] == 0]
    high_skew = [row["Category"] for row in report_data if float(row["Drop Rate"].replace("%","")) > 40.0]

    if completely_lost:
        print(f"CRITICAL ALERT: The following categories were 100% OMITTED:\n   {completely_lost}")
    if high_skew:
        print(f"SKEW WARNING: These categories lost over 40% of their total samples:\n   {high_skew}")
    if not completely_lost and not high_skew:
        print("DATA BALANCE SECURE: No critical omissions or extreme category skews found.")

if __name__ == "__main__":
    analyze_category_skew()