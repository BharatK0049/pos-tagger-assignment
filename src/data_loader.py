import nltk
from nltk.corpus import brown

def load_pos_data(split_ratio=0.8):
    """
    Fetches tagged sentences from the Brown corpus.
    
    Args:
        split_ratio (float): The percentage of data used for training.
        
    Returns:
        tuple: (train_sents, test_sents) where each is a list of 
               sentences containing (word, tag) tuples.
    """
    print("Downloading Brown corpus...")
    nltk.download('brown')
    nltk.download('universal_tagset')
    
    # We use the universal tagset to simplify the learning task initially
    # as mentioned in the assignment goals.
    tagged_sents = brown.tagged_sents(tagset='universal')
    
    split = int(len(tagged_sents) * split_ratio)
    train_data = tagged_sents[:split]
    test_data = tagged_sents[split:]
    
    print(f"Data Loaded: {len(train_data)} training sentences, {len(test_data)} test sentences.")
    return train_data, test_data