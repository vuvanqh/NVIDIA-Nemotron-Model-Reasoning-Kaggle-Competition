import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Set up matplotlib and seaborn style for a premium dark mode matching the presentation
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#000000',
    'axes.facecolor': '#101010',      # Matches slide card background
    'savefig.facecolor': '#000000',
    'text.color': '#edf2f7',
    'axes.labelcolor': '#edf2f7',
    'xtick.color': '#edf2f7',
    'ytick.color': '#edf2f7',
    'axes.edgecolor': '#1f1f1f',      # Matches slide card border
    'grid.color': '#222222',
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif'
})

# Custom NVIDIA color palette
NVIDIA_GREEN = '#76b900'
TEAL = '#2a9d8f'
SLATE = '#9aa7b8'
CORAL = '#e76f51'
PURPLE = '#8338ec'
BLUE = '#4a90e2'

CUSTOM_PALETTE = [NVIDIA_GREEN, TEAL, SLATE, CORAL, PURPLE, BLUE]

# Set working directory to the directory of this script or check paths
train_path = 'eda/train.csv'
test_path = 'eda/test.csv'

if not os.path.exists(train_path):
    # If running from inside eda/
    train_path = 'train.csv'
    test_path = 'test.csv'

print(f"Loading datasets from: {os.path.abspath(train_path)}")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 1. Class Distribution Plot
def classify_task(prompt):
    if "bit manipulation" in prompt:
        return "Bit Manipulation"
    elif "encryption rules" in prompt:
        return "Text Encryption"
    elif "numeral system" in prompt:
        return "Numeral Conversion"
    elif "unit conversion" in prompt:
        return "Unit Conversion"
    elif "applied to equations" in prompt:
        return "Equations & Symbolic"
    elif "gravitational constant" in prompt:
        return "Physics Gravity"
    else:
        return "Unknown"

train_df['task_type'] = train_df['prompt'].apply(classify_task)
test_df['task_type'] = test_df['prompt'].apply(classify_task)

plt.figure(figsize=(10, 5))
task_counts_df = pd.DataFrame({
    'Train Set': train_df['task_type'].value_counts(),
    'Test Set': test_df['task_type'].value_counts()
}).fillna(0)
task_counts_df.plot(kind='bar', color=[NVIDIA_GREEN, CORAL], edgecolor='#1f1f1f', width=0.8, ax=plt.gca())
plt.title("Task Type Frequency: Train vs Test Sets", color=NVIDIA_GREEN, fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Task Type", fontsize=11, labelpad=10)
plt.ylabel("Count", fontsize=11, labelpad=10)
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('eda/class_distribution.png', dpi=300, facecolor='#000000')
plt.close()
print("Saved eda/class_distribution.png")

# 2. Shot Count Distribution Plot
def extract_shot_count(prompt):
    lines = [line.strip() for line in prompt.split('\n') if line.strip()]
    operators = ['->', 'becomes', '=', 'For t =']
    example_count = 0
    for line in lines:
        is_example = any(op in line for op in operators)
        is_instruction = any(word in line.lower() for word in ["determine", "convert the following", "decrypt the following", "write the number", "now,"])
        if is_example and not is_instruction:
            example_count += 1
    return example_count

train_df['n_shots'] = train_df['prompt'].apply(extract_shot_count)

plt.figure(figsize=(10, 5))
sns.boxplot(x='task_type', y='n_shots', data=train_df, palette=CUSTOM_PALETTE)
plt.title("Few-Shot Example Count Distribution by Task Type", color=NVIDIA_GREEN, fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Task Type", fontsize=11, labelpad=10)
plt.ylabel("Number of Examples in Prompt", fontsize=11, labelpad=10)
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('eda/shot_count_distribution.png', dpi=300, facecolor='#000000')
plt.close()
print("Saved eda/shot_count_distribution.png")

# 3. Bit Manipulation Heatmap Plot (correlation)
bit_df = train_df[train_df['task_type'] == 'Bit Manipulation'].copy()

def parse_bit_examples(prompt):
    pattern = r'([01]{8})\s*->\s*([01]{8})'
    examples = re.findall(pattern, prompt)
    return [(np.array([int(c) for c in inp]), np.array([int(c) for c in out])) for inp, out in examples]

all_inputs = []
all_outputs = []
for idx, row in bit_df.head(200).iterrows():
    examples = parse_bit_examples(row['prompt'])
    for inp, out in examples:
        all_inputs.append(inp)
        all_outputs.append(out)
if all_inputs:
    X_bits = np.stack(all_inputs)
    Y_bits = np.stack(all_outputs)
    corr_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            corr_matrix[i, j] = np.corrcoef(X_bits[:, i], Y_bits[:, j])[0, 1]
            
    plt.figure(figsize=(8, 6.5))
    # Using a coolwarm map styled on dark theme
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                xticklabels=[f"Out {j}" for j in range(8)], yticklabels=[f"In {i}" for i in range(8)],
                cbar_kws={'label': 'Correlation Coefficient'}, annot_kws={'size': 9})
    plt.title("Correlation Heatmap: Input Bits vs Output Bits", color=NVIDIA_GREEN, fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('eda/bit_manipulation_linearity.png', dpi=300, facecolor='#000000')
    plt.close()
    print("Saved eda/bit_manipulation_linearity.png")

# 4. Answer Type Distribution Plot
def categorize_answer(ans):
    ans = str(ans).strip()
    if re.match(r'^[01]{8}$', ans):
        return "Binary (8-bit)"
    if re.match(r'^[IVXLCDMivxlcdm]+$', ans):
        return "Roman Numeral"
    try:
        float(ans)
        return "Numeric Float"
    except ValueError:
        pass
    if " " in ans:
        return "Text Sentence"
    return "Symbolic String"

train_df['answer_category'] = train_df['answer'].apply(categorize_answer)

plt.figure(figsize=(8.5, 5))
sns.countplot(x='answer_category', data=train_df, palette=CUSTOM_PALETTE, edgecolor='#1f1f1f')
plt.title("Global Answer Category Distribution", color=NVIDIA_GREEN, fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Category", fontsize=11, labelpad=10)
plt.ylabel("Count", fontsize=11, labelpad=10)
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('eda/answer_type_distribution.png', dpi=300, facecolor='#000000')
plt.close()
print("Saved eda/answer_type_distribution.png")

# 5. Token Count by Task Plot
subword_tokenizer = None
tokenizer_name = "Tiktoken (GPT-2 Encoding)"
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    subword_tokenizer = lambda text: enc.encode(text)
except ImportError:
    subword_tokenizer = lambda text: text.split()
    tokenizer_name = "Fallback Whitespace Tokenizer"

train_df['prompt_len'] = train_df['prompt'].apply(len)
train_df['token_count'] = train_df['prompt'].apply(lambda x: len(subword_tokenizer(x)))

plt.figure(figsize=(10, 5))
sns.violinplot(x='task_type', y='token_count', data=train_df, palette=CUSTOM_PALETTE, inner="quartile")
plt.title(f"Token Count Distribution by Task Type", color=NVIDIA_GREEN, fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Task Type", fontsize=11, labelpad=10)
plt.ylabel("Token Count (GPT-2)", fontsize=11, labelpad=10)
plt.xticks(rotation=30, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('eda/token_count_by_task.png', dpi=300, facecolor='#000000')
plt.close()
print("Saved eda/token_count_by_task.png")

print("All charts successfully generated in dark mode and saved to eda/ directory.")
