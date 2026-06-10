import json
import os
import pandas as pd
from transformers import AutoTokenizer
from huggingface_hub import login


HF_ACCESS_TOKEN = "my_token"
login(token=HF_ACCESS_TOKEN)


#INPUT_FILE = "flash_lite_raw_reasoning.jsonl"
# READY_TO_TRAIN_FILE = "nemotron_ready_to_train.jsonl"
# NEEDS_SHORTENING_FILE = "needs_gemma_shortening.jsonl"

# INPUT_FILE = "salvaged_flash_lite_raw_reasoning_trial2.jsonl" 
# READY_TO_TRAIN_FILE = "nemotron_ready_to_train_salvaged_trial2.jsonl"
# NEEDS_SHORTENING_FILE = "needs_shortening_salvaged_trial2.jsonl"

# INPUT_FILE = "needs_shortening_salvaged_trial2.jsonl"  
# READY_TO_TRAIN_FILE = "nemotron_ready_to_train_salvaged_trial2_corrected.jsonl"
# NEEDS_SHORTENING_FILE = "needs_shortening_salvaged_trial2_corrected.jsonl"
INPUT_FILE = "manunal.jsonl"  
READY_TO_TRAIN_FILE = "nemotron_ready_to_train_salvaged_trial2_corrected.jsonl"
NEEDS_SHORTENING_FILE = "needs_shortening_salvaged_trial2_corrected.jsonl"

tokenizer = AutoTokenizer.from_pretrained("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", trust_remote_code=True)


def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Cannot find your source file '{INPUT_FILE}'. Check the filename path.")
        return


    for f_path in [READY_TO_TRAIN_FILE, NEEDS_SHORTENING_FILE]:
        if os.path.exists(f_path):
            os.remove(f_path)

    total_rows = 0
    ready_count = 0
    shorten_count = 0
    
    lengths = []

    
    with open(INPUT_FILE, "r", encoding="utf-8") as infile:
        for line in infile:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                r_id = record["id"]
                puzzle = record["puzzle"].strip()
                target = str(record["target_answer"]).strip()
                raw_trace = record["raw_reasoning"].strip()
                
                prompt_input = f"[PUZZLE]: {puzzle}\n[TARGET ANSWER]: {target}"
                
                compliance_tail = f"\n\nTherefore, the final answer is \\boxed{{{target}}}"
                assistant_output = f"{raw_trace}{compliance_tail}"
    
                full_training_sequence = f"User: {prompt_input}\nAssistant: {assistant_output}"
                
                token_count = len(tokenizer.encode(full_training_sequence))
                lengths.append(token_count)
                total_rows += 1
                

                output_data = {
                    "id": r_id,
                    "puzzle": puzzle,
                    "target_answer": target,
                    "input": prompt_input,
                    "output": assistant_output,
                    "nemotron_tokens": token_count
                }
                

                if token_count <= 1024:
                    with open(READY_TO_TRAIN_FILE, "a", encoding="utf-8") as f_ready:
                        f_ready.write(json.dumps(output_data) + "\n")
                    ready_count += 1
                else:
                    output_data["raw_reasoning_to_shorten"] = raw_trace
                    with open(NEEDS_SHORTENING_FILE, "a", encoding="utf-8") as f_shorten:
                        f_shorten.write(json.dumps(output_data) + "\n")
                    shorten_count += 1

            except Exception as e:
                continue

    if total_rows == 0:
        print("Dataset structure could not be processed successfully.")
        return

    df_lens = pd.Series(lengths)

    print(f"Total Rows Evaluated:       {total_rows}")
    print(f"Average Token Length:       {int(df_lens.mean())} tokens")
    print(f"Max Token Length Found:     {df_lens.max()} tokens")
    print(f"READY TO TRAIN (≤1024):  {ready_count} ({ready_count/total_rows*100:.1f}%)")
    print(f"NEEDS SHORTENING (>1024): {shorten_count} ({shorten_count/total_rows*100:.1f}%)")

    print(f"\nSaved ready-to-train lines to: {READY_TO_TRAIN_FILE}")
    print(f"Saved oversized lines to:        {NEEDS_SHORTENING_FILE}")


if __name__ == "__main__":
    main()