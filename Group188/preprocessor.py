import re
import time
import pickle
import numpy as np
from bs4 import BeautifulSoup
from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer


class Preprocessor:

    def __init__(self):
        # Raw file
        self.train_file = "raw/train_tweets.txt"
        self.test_file = "raw/test_tweets_unlabeled.txt"
        self.train_label = "label/train_label.pkl"
        # Cleaned file
        self.train_file_cleaned = "data/train_tweets_cleaned.pkl"
        self.test_file_cleaned = "data/test_tweets_cleaned.pkl"
        self.total_file_cleaned = "data/total_tweets_cleaned.pkl"
        # Vector File
        self.train_vector = "vector/train_vector.npy"
        self.test_vector = "vector/test_vector.npy"
        # Model
        self.model = "model/model.bin"

    def clean(self):
        train_file_cleaned, test_file_cleaned, total_file_cleaned, train_label = [], [], [], []
        with open(self.train_file) as train_data:
            for line in train_data:
                label, tweet = line.strip().split('\t', 1)[:2]
                train_label.append(label)
                tweet_cleaned = self.tokenize(tweet)
                train_file_cleaned.append(tweet_cleaned)
                total_file_cleaned.append(tweet_cleaned)
        with open(self.test_file) as test_data:
            for line in test_data:
                tweet_cleaned = self.tokenize(line)
                test_file_cleaned.append(tweet_cleaned)
                total_file_cleaned.append(tweet_cleaned)
        pickle.dump(train_file_cleaned, open(self.train_file_cleaned, 'wb'))
        pickle.dump(test_file_cleaned, open(self.test_file_cleaned, 'wb'))
        pickle.dump(total_file_cleaned, open(self.total_file_cleaned, 'wb'))
        pickle.dump(train_label, open(self.train_label, 'wb'))

    @staticmethod
    def tokenize(text):
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9]+'
        pat2 = r'https?://[A-Za-z0-9./]+'
        combined_pat = r'|'.join((pat1, pat2))
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        clean = re.sub(combined_pat, '', souped)
        letters_only = re.sub("[^a-zA-Z]", " ", clean)
        lower_case = letters_only.lower()
        words = tok.tokenize(lower_case)
        # return " ".join(words)
        return words

    def train(self, dimension=64):
        start = time.time()
        print("Training Word2Vec model...")
        total_file_cleaned = pickle.load(open(self.total_file_cleaned, 'rb'))
        model = Word2Vec(total_file_cleaned, size=dimension, min_count=1)
        model.save(self.model)
        print("Model saved in : %s seconds " % (time.time() - start))

    def vectorize(self, dimension=64):
        model = Word2Vec.load(self.model)
        train_vector, test_vector = [], []
        count = 0
        for words in pickle.load(open(self.train_file_cleaned, 'rb')):
            vector = [0 for _ in range(dimension)]
            for word in words:
                vector += model[word]
            if len(words) != 0:
                vector /= len(words)
            train_vector.append(vector)
            count += 1
            if count % 10000 == 0:
                print("%s tweets vectorized..." % count)
        pickle.dump(np.array(train_vector), open(self.train_vector, 'wb'))
        for words in pickle.load(open(self.test_file_cleaned, 'rb')):
            vector = [0 for _ in range(dimension)]
            for word in words:
                vector += model[word]
            if len(words) != 0:
                vector /= len(words)
            test_vector.append(vector)
            count += 1
            if count % 10000 == 0:
                print("%s tweets vectorized..." % count)
        pickle.dump(np.array(test_vector), open(self.test_vector, 'wb'))

    def test(self):
        counter = set()
        for line in pickle.load(open(self.train_label, 'rb')):
            counter.add(line)
        print("Total label: ", len(counter))
        print("Train data: ", len(pickle.load(open(self.train_file_cleaned, 'rb'))))
        print("Test data: ", len(pickle.load(open(self.test_file_cleaned, 'rb'))))
        print("Total data: ", len(pickle.load(open(self.total_file_cleaned, 'rb'))))
        print("Train vector length: ", len(pickle.load(open(self.train_vector, 'rb'))))
        print("Test vector length: ", len(pickle.load(open(self.test_vector, 'rb'))))


preprocessor = Preprocessor()
preprocessor.clean()
preprocessor.train(dimension=128)
preprocessor.vectorize(dimension=128)
preprocessor.test()
