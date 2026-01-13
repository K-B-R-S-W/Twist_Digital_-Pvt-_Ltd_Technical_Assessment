# Twist_Digital
# Deceptive Review Detection - Technical Assessment

##  Overview
This repository contains my complete solution to the Twist Digital AI Engineer assessment. The system is designed to detect semantic inconsistencies, psychological manipulation, and stylistic fingerprinting in e-commerce reviews using a multi-branch NLP architecture.

##  Structure
- **`contradiction_detector.py`** - Implementation of the semantic contradiction detector (Part 2).
- **`part1_design.md`** - Multi-branch NLP pipeline design document (Part 1).
- **`part3.md`** - Zero-shot manipulation proposals document (Parts 3).
- **`part4.md`** - System design answers (Parts 4).
- **`dataset.txt`** - The provided test dataset.
- **`requirements.txt`** - Python dependencies required to run the code.

##  Important Note: Filename Change
### Why `contradiction_detector.py` instead of `code.py`?

**Reasoning:** The Python standard library contains a module named `code` (used by the debugger `pdb` and `torch`). Naming a script `code.py` in the root directory causes a circular import error (`ImportError: cannot import name 'CrossEncoder'`) when initializing PyTorch-based libraries. Renaming the file resolves this namespace conflict and ensures the code runs without crashing.

##  Important Note: Dataset Format Fix
### Changes Made to `dataset.txt`

The original `dataset.txt` file contained **Python-style boolean values** (`True`/`False`) and **tuple syntax** for spans, which are incompatible with JSON parsing:

**Issues Found:**
- `"has_contradiction": True` → Invalid JSON (requires lowercase `true`)
- `"contradiction_spans": [(0, 50), (51, 110)]` → Python tuples not valid JSON (requires arrays)

**Fixes Applied:**
- Changed all `True` → `true` and `False` → `false` (JSON standard)
- Converted tuples to arrays: `(0, 50)` → `[0, 50]` (JSON standard)

##  How to Run Part 2
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the detector:
   ```bash
   python contradiction_detector.py
   ```

**Note:** The first run will automatically download the NLI model (~499MB).

##  Key Highlights
- **100% Accuracy:** Achieved perfect Precision and Recall on the provided dataset.
- **Robustness:** Successfully handles challenging edge cases like Entity Resolution ("Camera good" vs "Battery bad"), Numeric Normalization, and Temporal Updates.
- **Architecture:** Implements the Multi-branch ensemble architecture designed in Part 1.
- **Optimization:** Includes production-ready strategies for latency (<50ms) and adversarial defense.

## Note on Implementation Logic
The code snippets provided in the responses below (Parts 3 & 4) are **simplified logic demonstrations**. They are intended to explicitly illustrate the specific libraries (Snorkel, SetFit, ONNX) and algorithmic strategies proposed, rather than serving as deployable production modules.

## 📬 Contact
Ravindu Sankalpa  
[k.b.ravindusankalpaac@gmail.com]
