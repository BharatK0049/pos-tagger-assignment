import pickle
from pathlib import Path
from data_loader import load_pos_data
from hmm_tagger import HMMTagger
from mem_tagger import MEMTagger
from tqdm import tqdm

def evaluate_hmm(tagger, test_data):
    correct, total = 0, 0
    for sentence in tqdm(test_data, desc="Evaluating HMM"):
        words = [w for w, t in sentence]
        true_tags = [t for w, t in sentence]
        preds = tagger.viterbi(words)
        for p, t in zip(preds, true_tags):
            if p == t: correct += 1
            total += 1
    return (correct / total) * 100

def evaluate_mem(tagger, test_data):
    correct, total = 0, 0
    for sentence in tqdm(test_data, desc="Evaluating MEM"):
        words = [w for w, t in sentence]
        true_tags = [t for w, t in sentence]
        preds = tagger.tag(words) # MEM predicts sentence at once
        for p, t in zip(preds, true_tags):
            if p == t: correct += 1
            total += 1
    return (correct / total) * 100

def main():
    train_sents, test_sents = load_pos_data(split_ratio=0.8)
    
    # Define file names for our saved models
    HMM_FILE = "hmm_model.pkl"
    MEM_FILE = "mem_model.pkl"

# --- HMM Section ---
    if Path(HMM_FILE).exists():
        print(f"\nLoading saved HMM from {HMM_FILE}...")
        with open(HMM_FILE, 'rb') as f:
            hmm = pickle.load(f)
    else:
        print("\n[1/2] Training HMM...")
        hmm = HMMTagger()
        hmm.train(train_sents)
        print(f"Saving HMM to {HMM_FILE}...")
        with open(HMM_FILE, 'wb') as f:
            pickle.dump(hmm, f)
    
    hmm_acc = evaluate_hmm(hmm, test_sents)

    # --- MEM Section ---
    if Path(MEM_FILE).exists():
        print(f"\nLoading saved MEM from {MEM_FILE}...")
        with open(MEM_FILE, 'rb') as f:
            mem = pickle.load(f)
    else:
        print("\n[2/2] Training MEM (Logistic Regression)...")
        mem = MEMTagger()
        mem.train(train_sents)
        print(f"Saving MEM to {MEM_FILE}...")
        with open(MEM_FILE, 'wb') as f:
            pickle.dump(mem, f)
            
    mem_acc = evaluate_mem(mem, test_sents)
    
    # --- FINAL COMPARISON ---
    print("\n" + "="*30)
    print(f"HMM Accuracy: {hmm_acc:.2f}%")
    print(f"MEM Accuracy: {mem_acc:.2f}%")
    print("="*30)
    
    # Demonstration of the "Rule-Based" fix mentioned in notes
    # Example: "the red investigation" (ADJ between DET and NOUN)
    sample = "The red investigation".split()
    print(f"\nLive Test on: {sample}")
    print(f"HMM Tags: {hmm.viterbi(sample)}")
    print(f"MEM Tags: {mem.tag(sample)}")

if __name__ == "__main__":
    main()