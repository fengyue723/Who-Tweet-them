import re, time, pickle, random
import pandas as pd
import numpy as np
from scipy import sparse
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

random.seed(895376)
sample = 0.001  # 0.1%


class Pipeline:

    def __init__(self):
        # Vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        # Classifier
        self.classifier = OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=1)
        # self.classifier = OneVsRestClassifier(LinearSVC(), n_jobs=1)
        # self.classifier = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))
        # Raw file
        self.train_file = "raw/train_tweets.txt"
        self.test_file = "raw/test_tweets_unlabeled.txt"
        # Cleaned file
        self.train_file_cleaned = "data/train_tweets_cleaned.txt"
        self.test_file_cleaned = "data/test_tweets_cleaned.txt"
        self.total_file_cleaned = "data/total_tweets_cleaned.txt"
        # Vector File
        self.train_vector = "vector/train.vec"
        self.test_vector = "vector/test.vec"
        # Label File
        self.train_label = "label/train_label.txt"
        self.test_label = "label/test_label.csv"

    def clean(self):
        train_file_cleaned = open(self.train_file_cleaned, 'w')
        test_file_cleaned = open(self.test_file_cleaned, 'w')
        total_file_cleaned = open(self.total_file_cleaned, 'w')
        train_label = open(self.train_label, 'w')
        with open(self.train_file) as train_data:
            for line in train_data:
                label, tweet = line.strip().split('\t', 1)[:2]
                train_label.write(label + '\n')
                tokenized_tweet = self.tokenize(tweet)
                train_file_cleaned.write(tokenized_tweet + '\n')
                total_file_cleaned.write(tokenized_tweet + '\n')
        with open(self.test_file) as test_data:
            for line in test_data:
                tokenized_tweet = self.tokenize(line)
                test_file_cleaned.write(tokenized_tweet + '\n')
                total_file_cleaned.write(tokenized_tweet + '\n')

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
        return " ".join(words)

    def vectorize(self):
        total_file_cleaned = open(self.total_file_cleaned)
        print("Fitting vectorizer...")
        self.vectorizer.fit(total_file_cleaned)
        print("Vectorizing train file...")
        train_file_cleaned = open(self.train_file_cleaned)
        train_vector = self.vectorizer.transform(train_file_cleaned)
        print("Train vector: ", train_vector.shape)
        print("Vectorizing test file...")
        test_file_cleaned = open(self.test_file_cleaned)
        test_vector = self.vectorizer.transform(test_file_cleaned)
        print("Test vector: ", test_vector.shape)
        print("Saving...")
        pickle.dump(train_vector, open(self.train_vector, 'wb'))
        pickle.dump(test_vector, open(self.test_vector, 'wb'))

    def evaluate(self, sampling=False):
        train_vector = pickle.load(open(self.train_vector, 'rb'))
        train_label = []
        with open(self.train_label) as file:
            for line in file:
                train_label.append(int(line))
        if sampling:
            _, train_vector, _, train_label = train_test_split(train_vector, train_label, test_size=sample)
        print("Data: ", train_vector.shape)
        X_train, X_evl, y_train, y_evl = train_test_split(train_vector, train_label, test_size=0.1, random_state=0)
        start = time.time()
        print("Training Classifier...")
        self.classifier.fit(X_train, y_train)
        print("Training successfully in: %s seconds " % (time.time() - start))
        print("Evaluating...")
        pred_labels = self.classifier.predict(X_evl)
        print("Evaluate Accuracy: %s" % (accuracy_score(y_evl, pred_labels)))

    def classify(self):
        print("Predicting...")
        test_vector = pickle.load(open(self.test_vector, 'rb'))
        test_label = self.classifier.predict(test_vector)
        df = pd.DataFrame(test_label, columns=['Predicted'])
        df.index += 1
        df.index.name = 'Id'
        df.to_csv(self.test_label)


pipe = Pipeline()
# pipe.clean()
# pipe.vectorize()
pipe.evaluate(sampling=True)
pipe.classify()
