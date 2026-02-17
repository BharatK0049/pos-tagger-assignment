import numpy as np
from collections import defaultdict, Counter

class HMMTagger:
    """
    Hidden Markov Model for POS Tagging.
    Calculates P(Tag|Prev_Tag) and P(Word|Tag).
    """
    def __init__(self):
        # A: Transition Probabilities {prev_tag: {current_tag: count}}
        self.transitions = defaultdict(Counter)
        # B: Emission Probabilities {tag: {word: count}}
        self.emissions = defaultdict(Counter)
        # Unique tags in the corpus
        self.tags = set()

    def train(self, tagged_sentences):
        """
        Learns transition and emission counts from tagged data.
        """
        for sentence in tagged_sentences:
            prev_tag = "START"  # Sentence boundary marker
            self.tags.add(prev_tag)
            
            for word, tag in sentence:
                word = word.lower()
                self.tags.add(tag)
                
                # Transition: How likely is this tag after the previous one?
                self.transitions[prev_tag][tag] += 1
                
                # Emission: How likely is this tag to produce this word?
                self.emissions[tag][word] += 1
                
                prev_tag = tag

    def get_transition_prob(self, prev_tag, current_tag):
        """
        Calculates P(tag | prev_tag). 
        Formula: (count + 1) / (total_transitions_from_prev + total_unique_tags)
        """
        total_transitions_from_prev = sum(self.transitions[prev_tag].values())

        # Count how many times it transitioned specifically to current_tag
        count = self.transitions[prev_tag][current_tag]
        
        # V = number of unique tags in our vocabulary
        v = len(self.tags)
        
        # Apply Add-1 Smoothing
        return (count + 1) / (total_transitions_from_prev + v)

    def get_emission_prob(self, tag, word):
        """
        Calculates P(word | tag).
        Formula: (count + 1) / (total_emissions_for_tag + total_unique_words)
        """
        # We need a set of all unique words seen in training for the denominator
        if not hasattr(self, 'vocab_size'):
            unique_words = set()
            for word_counts in self.emissions.values():
                unique_words.update(word_counts.keys())
            self.vocab_size = len(unique_words)

        total_emissions_for_tag = sum(self.emissions[tag].values())
        count = self.emissions[tag][word.lower()]
        
        # Apply Add-1 Smoothing
        return (count + 1) / (total_emissions_for_tag + self.vocab_size)
    
    def viterbi(self, sentence: list) -> list:
        """
        Finds the most likely sequence of POS tags for a given sentence.
        
        Args:
            sentence (list): A list of word strings.
            
        Returns:
            list: The most probable sequence of tags.
        """
        # Convert sentence to lowercase to match training data
        words = [w.lower() for w in sentence]
        
        # We exclude "START" from our candidate tags for the trellis
        states = [t for t in self.tags if t != "START"]
        num_states = len(states)
        num_words = len(words)
        
        # viterbi_matrix[i][j] = max probability of being in state 'i' at word 'j'
        # backpointer[i][j] = the previous state that led to this max probability
        viterbi_matrix = np.zeros((num_states, num_words))
        backpointer = np.zeros((num_states, num_words), dtype=int)
        
        # 1. Initialization Step (First Word)
        for s in range(num_states):
            tag = states[s]
            # P(tag | START) * P(word | tag)
            viterbi_matrix[s, 0] = self.get_transition_prob("START", tag) * \
                                   self.get_emission_prob(tag, words[0])
            backpointer[s, 0] = 0
            
        # 2. Recursion Step (Words 2 to N)
        for w in range(1, num_words):
            for s in range(num_states):
                current_tag = states[s]
                
                # Calculate probabilities for all possible previous tags
                # P(prev_state) * P(current_tag | prev_tag) * P(word | current_tag)
                probs = [viterbi_matrix[prev_s, w-1] * self.get_transition_prob(states[prev_s], current_tag) * self.get_emission_prob(current_tag, words[w])
                         for prev_s in range(num_states)]
                
                viterbi_matrix[s, w] = max(probs)
                backpointer[s, w] = np.argmax(probs)
                
        # 3. Path Backtracking
        best_path = []
        # Find the most likely final state
        best_last_state = np.argmax(viterbi_matrix[:, num_words - 1])
        best_path.append(states[best_last_state])
        
        # Follow the backpointers to the beginning
        curr_state_idx = best_last_state
        for w in range(num_words - 1, 0, -1):
            curr_state_idx = backpointer[curr_state_idx, w]
            best_path.insert(0, states[curr_state_idx])
            
        return best_path