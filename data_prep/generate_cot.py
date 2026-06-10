import os
import asyncio
import json
import pandas as pd
from google import genai
from google.genai import errors

#
#INPUT_CSV = "train.csv"
INPUT_CSV = "salvaged_puzzles_trial2.csv"  
#OUTPUT_JSONL = "flash_lite_raw_reasoning.jsonl"
OUTPUT_JSONL = "salvaged_flash_lite_raw_reasoning_trial2.jsonl"  

API_KEYS = []


io_lock = asyncio.Lock()

class KeyWorker:
    def __init__(self, api_key):
        self.client = genai.Client(api_key=api_key)
        self.masked_key = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "UnknownKey"
        self.rpm_delay = 60.0 / 15  
        self.is_exhausted = False
        self.last_request_time = 0

    async def wait_for_turn(self):
        now = asyncio.get_event_loop().time()
        elapsed = now - self.last_request_time
        if elapsed < self.rpm_delay:
            await asyncio.sleep(self.rpm_delay - elapsed)
        self.last_request_time = asyncio.get_event_loop().time()

def get_processed_ids():
    processed = set()
    if os.path.exists(OUTPUT_JSONL):
        with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        processed.add(str(data["id"]))
                    except:
                        continue
    return processed

async def write_success_record(data):
    async with io_lock:
        with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(data) + "\n")

async def worker_task(worker: KeyWorker, task_queue: asyncio.Queue):
    # system_instruction = (
    #     "You are an ultra-precise logical reasoning engine.\n"
    #     "Solve the puzzle step by step. Explain your logical deductions completely.\n"
    #     "Ensure your final conclusion perfectly matches the target correct answer provided."
    # )
    # system_instruction = (
    #     "You are an ultra-precise, hyper-concise logical reasoning engine.\n"
    #     "Solve the puzzle step by step using compact mathematical or symbolic notation.\n"
    #     "CRITICAL CONSTRAINTS:\n"
    #     "- Do not include any introductory fluff, conversational filler, or boilerplate text.\n"
    #     "- Condense your reasoning logic into exactly 2 or 3 brief, high-density bullet points.\n"
    #     "- Show the core transformations or calculations directly. Do not use verbose prose.\n"
    #     "- STOP immediately after the final calculation step. DO NOT write a summary statement, "
    #     "conclusion sentence, or final answer phrase."
    # )

    system_instruction = (
        "You are an expert mathematical proof condenser. You do not solve puzzles; you write "
        "flawless, direct verification traces.\n"
        "CRITICAL CONSTRAINTS:\n"
        "- You are forbidden from making mistakes, backpedaling, or correcting yourself.\n"
        "- Do not explore false paths. Write ONLY the final, correct, direct line of logic.\n"
        "- Assume your first thought is perfectly correct. Express it immediately using "
        "dense mathematical transitions (e.g., A -> B -> C).\n"
        "- Absolutely no conversational self-talk or meta-commentary."
    )

    while not task_queue.empty() and not worker.is_exhausted:
        try:
            puzzle_id, puzzle_text, target_answer = await task_queue.get()
        except asyncio.QueueEmpty:
            break

        await worker.wait_for_turn()

       # user_content = f"Puzzle:\n{puzzle_text}\n\nTarget Correct Answer:\n{target_answer}"
        # user_content = (
        #     f"Puzzle:\n{puzzle_text}\n\n"
        #     f"Target Correct Answer:\n{target_answer}\n\n"
        #     f"CRITICAL: Provide only the dense, step-by-step logic required to reach this target answer. "
        #     f"Stop immediately when the final math calculation is shown."
        # )
        
        user_content = (
            f"Puzzle:\n{puzzle_text}\n\n"
            f"Known Final Answer:\n{target_answer}\n\n"
            f"INSTRUCTION:\n"
            f"Reverse-engineer the exact, clean logic that leads directly from the puzzle state "
            f"to the Known Final Answer. Since you already know the destination, do not guess, "
            f"do not self-correct, and do not write out any failed attempts. Provide ONLY the "
            f"smooth, uninterrupted 2-step symbolic transformation required."
        )
        backoff = 3
        success = False

        while not success and not worker.is_exhausted:
            try:
                response = await worker.client.aio.models.generate_content(
                    model='gemini-3.1-flash-lite',
                    contents=user_content,
                    config={
                        "system_instruction": system_instruction,
                        #"temperature": 0.0,
                        "temperature": 0.2, #higher temperature to encourage more flexible reasoning paths for the failed puzzles (trial2) 
                        "max_output_tokens": 1200
                    }
                )
                
                generated_text = response.text if response.text else ""
                if not generated_text.strip():
                    await asyncio.sleep(2)
                    continue

                output_record = {
                    "id": str(puzzle_id),
                    "puzzle": puzzle_text,
                    "target_answer": target_answer,
                    "raw_reasoning": generated_text
                }
                
                await write_success_record(output_record)
                print(f"✔️ Key [{worker.masked_key}] | Row ID {puzzle_id} generated successfully.")
                success = True
                task_queue.task_done()

            except errors.APIError as e:
                err_msg = str(e.message).lower()
                if e.code == 429:
                    if "daily" in err_msg or "today" in err_msg:
                        print(f"Key [{worker.masked_key}] hit hard daily quota error. Stopping worker.")
                        worker.is_exhausted = True
                        await task_queue.put((puzzle_id, puzzle_text, target_answer))
                        task_queue.task_done()
                        break
                    else:
                        print(f"Key [{worker.masked_key}] hit soft limit. Re-queueing row and backing off {backoff}s...")
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 30)
                        await task_queue.put((puzzle_id, puzzle_text, target_answer))
                        task_queue.task_done()
                        success = True
                else:
                    print(f"API Error on Row ID {puzzle_id} via Key [{worker.masked_key}]: {e}")
                    await task_queue.put((puzzle_id, puzzle_text, target_answer))
                    task_queue.task_done()
                    success = True 

            except Exception as e:
                print(f"Unexpected error on Row ID {puzzle_id}: {e}")
                await task_queue.put((puzzle_id, puzzle_text, target_answer))
                task_queue.task_done()
                success = True

async def main_async():
    df = pd.read_csv(INPUT_CSV)
    processed_ids = get_processed_ids()
    
    task_queue = asyncio.Queue()
    tasks_count = 0
    for _, row in df.iterrows():
        r_id = str(row['id'])
        if r_id not in processed_ids:
            await task_queue.put((r_id, row['prompt'], row['answer']))
            tasks_count += 1
            
    if tasks_count == 0:
        print("Complete! All rows from the CSV have already been processed.")
        return
        
    print(f"Starting Stage 1 Execution. Total CSV: {len(df)} | Remaining: {tasks_count}")
    
    workers = [KeyWorker(key) for key in API_KEYS]
    worker_loops = [asyncio.create_task(worker_task(w, task_queue)) for w in workers]
    
    await asyncio.gather(*worker_loops)
    print("Stage 1 complete or all keys exhausted.")

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main()
    
    
