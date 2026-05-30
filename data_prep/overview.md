I’ve finalized the baseline version of our dataset with CoT and uploaded it to the Kaggle dataset that I shared with the group I created, as well as to GitHub for easy modification. You should have full access to both the Kaggle dataset and the Kaggle notebook now - it's a copy of a demo notebook, you don't have to use it, I didn't run it.

### Dataset Distribution & Strategy
* **Stratified 85/15 Split:** The train/validation split was executed as a stratified split by category (`task_type`). The python code used for the classification, formatting, and splitting can be found in `prepare_competition_v1.py`.
* **Quite Balanced Classes:** There are at least 1,000 puzzles for every category.
* **No Synthetic Augmentation:** I decided against running synthetic data generation for this version because the original class split was already perfectly fair and balanced. I thought it would make more sense for us to hold off on augmentation until we run initial evaluations and see if the model is heavily failing on one specific category.

### Structure & Token Constraints
* **Pruned & Modular Schema:** The final files are structured as modular `.jsonl` objects. I mapped everything to fields (`id`, `task_type`, `puzzle`, `target_answer`, `output`, `nemotron_tokens`) so it’s easy for anyone to read or modify.
* **1024 Token Ceiling:** Nemotron tokenizer counts were validated using the sequence layout: `User: [PUZZLE]: ... \nAssistant: ...`. All of the entries in these files fit under 1024 tokens. Hovewer, this sequence is no longer in the dataset, so if you will use a different there might be an issue.

### V1
* **The 15% Salvage Queue:** The Chain-of-Thought (CoT) traces were generated using `gemini-3.1-flash-lite`. About 15% of the raw generations failed/cut off because the native Gemini sequences ran too long. I am currently re-running and compressing this chunk locally; once done, I’ll attach them to a new version of the dataset.

### V2
whole dataset (without 2 rows) is now included

* **Messy Source Files:** I will later push more of the raw intermediate files to GitHub just in case anyone wants them, but they are incredibly messy right now, so I wouldn't recommend reading through them unless you absolutely have to.

