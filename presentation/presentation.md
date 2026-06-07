---
marp: true
theme: default
paginate: true
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

  section {
    background: #000000;
    color: #edf2f7;
    font-family: 'Inter', system-ui, sans-serif;
    padding: 50px 70px;
  }

  section.plot {
    padding: 42px 55px 40px;
  }

  h1 {
    color: #76b900;
    font-size: 1.76em;
    font-weight: 800;
    margin: 0 0 6px;
    line-height: 1.15;
  }

  section.plot h1 {
    margin-bottom: 18px;
  }

  h2 {
    color: #9aa7b8;
    font-size: 0.68em;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    border: none;
    margin: 0 0 22px;
  }

  h3 {
    color: #edf2f7;
    font-size: 0.92em;
    margin: 0 0 14px;
  }

  ul {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  li {
    font-size: 0.88em;
    line-height: 1.48;
    margin-bottom: 15px;
    padding-left: 18px;
    position: relative;
  }

  li::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0.63em;
    width: 5px;
    height: 5px;
    border-radius: 50%;
    background: #76b900;
  }

  strong {
    color: #76b900;
  }

  code {
    background: #171717;
    color: #a3c97a;
    padding: 1px 7px;
    border-radius: 4px;
    font-size: 0.86em;
  }

  section::after {
    color: #2f2f2f;
    font-size: 0.65em;
    bottom: 24px;
    right: 70px;
  }

  section.plot::after {
    right: 55px;
    bottom: 22px;
  }

  .cols {
    display: grid;
    grid-template-columns: 0.95fr 1.05fr;
    gap: 42px;
    align-items: center;
  }

  .cols.reverse {
    grid-template-columns: 1.05fr 0.95fr;
  }

  .cols img {
    width: 100%;
    max-height: 405px;
    height: auto;
    object-fit: contain;
    border-radius: 0;
    opacity: 1;
    box-shadow: none;
  }

  .note {
    color: #a7b4c4;
    font-size: 0.72em;
    line-height: 1.5;
    margin-top: 22px;
    border-left: 2px solid #76b900;
    padding-left: 14px;
  }

  section.cover {
    background: #000000;
    padding-bottom: 78px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
  }

  section.cover .label {
    font-size: 0.68em;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #76b900;
    font-weight: 700;
    margin-bottom: 18px;
  }

  section.cover h1 {
    font-size: 2.55em;
    line-height: 1.08;
    margin-bottom: 16px;
  }

  section.cover .sub {
    color: #c8d0dc;
    font-size: 0.88em;
    line-height: 1.6;
    margin-bottom: 40px;
    max-width: 58%;
  }

  section.cover .meta {
    display: flex;
    gap: 40px;
    border-top: 1px solid rgba(118,185,0,0.28);
    padding-top: 22px;
    font-size: 0.74em;
    color: #9aa7b8;
  }

  section.cover .meta b {
    display: block;
    font-size: 1.18em;
    color: #edf2f7;
    font-weight: 700;
    margin-bottom: 2px;
  }

  .metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
    margin-top: 22px;
  }

  .card {
    background: #101010;
    border: 1px solid #1f1f1f;
    border-radius: 12px;
    padding: 20px;
  }

  .value {
    font-size: 1.95em;
    font-weight: 800;
    color: #76b900;
    line-height: 1;
  }

  .label {
    margin-top: 10px;
    color: #9aa7b8;
    font-size: 0.70em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .fullplot {
    height: 520px;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .fullplot img {
    max-width: 98%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
    border-radius: 0;
    box-shadow: none;
    opacity: 1;
  }

  .fullplot-wide {
    height: 505px;
    display: flex;
    justify-content: center;
    align-items: center;
  }

  .fullplot-wide img {
    max-width: 96%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    display: block;
    border-radius: 0;
    box-shadow: none;
    opacity: 1;
  }
---

<!-- _class: cover -->
<!-- _paginate: false -->

<div class="label">Kaggle · NVIDIA · 2026</div>

# NVIDIA Nemotron<br>Model Reasoning<br>Challenge

<div class="sub">
Team Moroco
</div>

<div class="meta">
  <span><b>Hosted by</b>NVIDIA on Kaggle</span>
  <span><b>Model</b>Nemotron-3-Nano-30B</span>
  <span><b>Method</b>LoRA adapter fine-tuning</span>
</div>

---

# Teaching a Giant to Think

## What this competition is

<div class="cols">
<div>

- NVIDIA provides a **frozen 30B open model**; participants improve its reasoning behavior
- The allowed training mechanism is a lightweight **LoRA adapter**
- The adapter is merged with the base model at **evaluation time via vLLM**
- Success depends on generalisation to **unseen puzzle types**, not memorisation

</div>

![Competition overview](images/intro_images/slide1.png)

</div>

---

# The Problem

## What the model must solve

<div class="cols reverse">

![Reasoning puzzle illustration](images/intro_images/slide2.png)

<div>

- Each task provides **input–output examples** that imply a hidden rule
- The model must **infer the transformation** and apply it to a new case
- Puzzle families include **bit manipulation**, **algebraic equations**, and **abstract sequences**
- The final answer must be wrapped in `\boxed{}` and is scored by **exact match accuracy**

</div>

</div>

---

# Limitations & Constraints

## What is and is not permitted

<div class="cols">
<div>

- LoRA rank is **capped at 32**, so the adapter capacity is intentionally limited
- Base model weights are **immutable**; full fine-tuning is not allowed
- Inference runs under a **fixed compute budget** on Kaggle infrastructure
- The private test set is **fully held out**, so the public leaderboard may overestimate progress

</div>

![Limitations and constraints illustration](images/intro_images/slide4.png)

</div>

---

# Exploratory Data Analysis

## Dataset quality and structure

<div class="metrics">

<div class="card">
<div class="value">6</div>
<div class="label">Task Categories</div>
</div>

<div class="card">
<div class="value">9500</div>
<div class="label">Training Samples</div>
</div>

<div class="card">
<div class="value">3</div>
<div class="label">Test Samples</div>
</div>

<div class="card">
<div class="value">0</div>
<div class="label">Duplicate Prompts or Missing Values</div>
</div>

<div class="card">
<div class="value">302</div>
<div class="label">Avg Train Length</div>
</div>

<div class="card">
<div class="value">472</div>
<div class="label">Avg Test Length</div>
</div>

</div>

<div class="note">
The dataset is clean, balanced, and requires virtually no preprocessing before fine-tuning.
</div>

---

# Task Taxonomy

<div class="fullplot">
  <img src="images/eda_images/class_distribution.png" />
</div>

---

# Few-Shot Prompt Structure

<div class="fullplot">
  <img src="images/eda_images/shot_count_distribution.png" />
</div>

---

# Bit Manipulation Deep Dive

<div class="fullplot">
  <img src="images/eda_images/bit_manipulation_linearity.png" />
</div>

---

# Answer Format Distribution

<div class="fullplot-wide">
  <img src="images/eda_images/answer_type_distribution.png" />
</div>

---

# Tokenization Analysis

<div class="fullplot-wide">
  <img src="images/eda_images/token_count_by_task.png" />
</div>
