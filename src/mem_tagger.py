from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

class MEMFeatureExtractor:
    def extract_features(self, sentence, index):
        """
        Extracts features for a word at a specific index in a sentence.
        
        Args:
            sentence (list): List of words in the sentence.
            index (int): The position of the word we are currently tagging.
        """
        word = sentence[index]
        prev_word = sentence[index - 1] if index > 0 else "<START>"
        next_word = sentence[index + 1] if index < len(sentence) - 1 else "<END>"

        features = {
            'bias': 1.0,
            'word': word.lower(),
            'is_first': index == 0,
            'is_last': index == len(sentence) - 1,
            'is_capitalized': word[0].isupper(),
            'is_all_caps': word.isupper(),
            'is_all_lower': word.islower(),
            'prefix-1': word[0],
            'prefix-2': word[:2],
            'suffix-1': word[-1],
            'suffix-2': word[-2:],
            'prev_word': prev_word.lower(),
            'next_word': next_word.lower(),
            'has_hyphen': '-' in word,
            'is_numeric': word.isdigit(),
        }
        return features

class MEMTagger:
    def __init__(self):
        # We use a Pipeline to handle feature vectorization and 
        # Logistic Regression (Maximum Entropy) in one step.
        self.model = Pipeline([
            ('vectorizer', DictVectorizer(sparse=True)),
            ('classifier', LogisticRegression( 
                solver='lbfgs', 
                max_iter=100, 
                verbose=1
            ))
        ])
        self.feature_extractor = MEMFeatureExtractor()

    def prepare_data(self, tagged_sentences):
        X = [] # Features
        y = [] # Labels (Tags)

        for sentence in tagged_sentences:
            words = [w for w, t in sentence]
            for i in range(len(sentence)):
                X.append(self.feature_extractor.extract_features(words, i))
                y.append(sentence[i][1])
        return X, y

    def train(self, tagged_sentences):
        print("Extracting features and training Maximum Entropy Model...")
        X, y = self.prepare_data(tagged_sentences)
        self.model.fit(X, y)

    def tag(self, sentence):
        """
        Tags a single sentence (list of words).
        """
        features = [self.feature_extractor.extract_features(sentence, i) 
                    for i in range(len(sentence))]
        return self.model.predict(features)