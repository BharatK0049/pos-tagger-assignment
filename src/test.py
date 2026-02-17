import pickle
from pathlib import Path

# Load the saved brains
with open("hmm_model.pkl", 'rb') as f:
    hmm = pickle.load(f)
with open("mem_model.pkl", 'rb') as f:
    mem = pickle.load(f)

def test_sentence(text):
    words = text.split()
    print(f"\nSentence: {text}")
    print(f"HMM Tags: {hmm.viterbi(words)}")
    print(f"MEM Tags: {mem.tag(words)}")

# Try these tricky ones!
test_sentence("The old man the boats")
test_sentence("I saw the man with the telescope")
test_sentence("The complex houses married and single soldiers")