# POS Tagging & Sequence Models: HMM vs. MEM

## Overview
This project implements and compares two fundamental approaches to Part-of-Speech (POS) tagging using the **Brown Corpus**:
1.  **Hidden Markov Model (HMM)**: A generative sequence model using the Viterbi algorithm.
2.  **Maximum Entropy Model (MEM)**: A discriminative model utilizing feature functions $f(x, y)$ and Logistic Regression.

The goal is to demonstrate how statistical models overcome the limitations of rule-based tagging (e.g., "words following 'the' are always nouns").

---

## 1. Hidden Markov Model (HMM)
The HMM identifies the correct tag by calculating the most likely sequence of hidden states (tags) for a sequence of observations (words).

### Core Components
* **Transition Probabilities ($A$):** $P(tag_i | tag_{i-1})$ — Measures the likelihood of one tag following another.
* **Emission Probabilities ($B$):** $P(word_i | tag_i)$ — Measures the likelihood of a tag producing a specific word.
* **Laplace (Add-1) Smoothing:** Applied to both matrices to handle Out-of-Vocabulary (OOV) words and unseen transitions, ensuring no zero-probability paths.
* **Viterbi Algorithm:** A dynamic programming approach used to decode the most probable "path" of tags for an entire sentence.

### Performance
* **Accuracy:** 92.69% 
* **Strength:** Excellent at capturing the global structure of English sentences.

---

## 2. Maximum Entropy Model (MEM)
The MEM is a discriminative model ($P(y|x)$) that allows for the integration of complex, non-independent features of the context.

### Feature Functions $f(x, y)$
Unlike the HMM, which is restricted to local transitions, the MEM uses a rich feature set to represent the context $x$:
* **Lexical Features:** The word itself (lowercase).
* **Morphological Features:** Suffixes (last 1–2 chars), prefixes, and capitalization patterns (e.g., `is_capitalized`, `has_hyphen`).
* **Contextual Features:** Identity of the `prev_word` and `next_word`.
* **Bias:** A base feature to represent the prior probability of tags.

### Key Advantage
MEM solves issues where rule-based logic fails by weighing competing features. For example, in the phrase *"The red investigation"*, the feature `prev_word='the'` might suggest a Noun, but the word `red` and its context eventually allow the model to correctly assign the `ADJ` tag.

---

## Project Structure
```text
pos_tagger_assignment/
├── data/                    # Local storage for NLTK corpus
├── src/
│   ├── data_loader.py       # Fetches (word, tag) pairs from Brown Corpus
│   ├── hmm_tagger.py        # HMM training and Viterbi decoding
│   ├── mem_tagger.py        # Feature extraction and Log-Reg training
│   └── main.py              # Comparative evaluation and live testing
├── hmm_model.pkl            # Serialized HMM (optional/generated)
├── mem_model.pkl            # Serialized MEM (optional/generated)
├── requirements.txt
└── README.md
```

## Installation and Usage

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Run Evaluation

```bash
python src/main.py
```

## Results and Comparison

| **Model** | **Accuracy** | **Logic** | **Handling of OOV Words** |
|-----------|-------------|------------|----------------------------|
| HMM       | 92.70%      | Generative (Transition / Emission probabilities) | Relies on smoothed emission probabilities |
| MEM       | 96.08%      | Discriminative (Feature-based approach) | Uses morphological features such as suffixes and capitalization |

## Output
<img width="425" height="74" alt="image" src="https://github.com/user-attachments/assets/b0255e48-756d-4711-95ba-efe23b0aca1a" />

<img width="584" height="253" alt="image" src="https://github.com/user-attachments/assets/727302cc-ab25-4524-901a-712033bb2240" />



## Key Observations
The Accuracy Gap: The MEM outperformed the HMM by 3.38%. This is primarily because the MEM can use morphological features (like the "-ing" or "-ed" suffixes) to "detect" the tag of a word it has never seen before, whereas the HMM has to rely mostly on the surrounding tags.

Efficiency Trade-off: While the HMM trains nearly instantaneously, its evaluation using the Viterbi algorithm is computationally heavy. The MEM takes much longer to train initially (as it solves an optimization problem) but offers much faster inference speeds once trained.
