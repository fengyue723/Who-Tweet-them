import re, time, pickle
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import *



seed = 895376
sample = 0.05  # 5%


class Pipeline:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                                          decode_error='strict', strip_accents=None, lowercase=True,
                                          preprocessor=None, tokenizer=None, analyzer='word',
                                          stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                                          ngram_range=(1, 1), max_df=1.0, min_df=1,
                                          max_features=50000, vocabulary=None, binary=False,
                                          dtype=np.float64, norm='l2')
        # self.classifier = LogisticRegression(penalty='l2', dual=False, tol=1e-4, C=1.0,
        #                                      fit_intercept=True, intercept_scaling=1, class_weight='balanced',
        #                                      random_state=seed, solver='sag', max_iter=100,
        #                                      multi_class='multinomial', n_jobs=4)
        self.classifier = LinearSVC(penalty='l2', loss='squared_hinge', dual=True,\
                                    tol=0.0001, C=1.0, multi_class='ovr',\
                                    fit_intercept=True, intercept_scaling=1,\
                                    class_weight=None, verbose=0,\
                                    random_state=seed, max_iter=1000)
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
        # pat1 = r'@[A-Za-z0-9]+'
        # pat2 = r'https?://[A-Za-z0-9./]+'
        # combined_pat = r'|'.join((pat1, pat2))
        # soup = BeautifulSoup(text, 'lxml')
        # souped = soup.get_text()
        # clean = re.sub(combined_pat, '', souped)
        # letters_only = re.sub("[^a-zA-Z]", " ", clean)
        # lower_case = letters_only.lower()
        words = tok.tokenize(text)
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

        train_label = []
        with open(self.train_label) as file:
            for line in file:
                train_label.append(int(line))

        # print("Feature selecting...")
        # _, train_vector_1, _, train_label_1 = train_test_split(train_vector, train_label, test_size=sample,
        #                                                        random_state=seed)
        # feature_select = SelectKBest(chi2, k=50000)
        # feature_select.fit(train_vector_1, train_label_1)
        # train_vector = feature_select.transform(train_vector)
        # test_vector = feature_select.transform(test_vector)
        # print("Train vector: ", train_vector.shape)
        # print("Test vector: ", test_vector.shape)
        # print("Saving...")

        pickle.dump(train_vector, open(self.train_vector, 'wb'))
        pickle.dump(test_vector, open(self.test_vector, 'wb'))

    def evaluate(self, sampling=False):
        train_vector = pickle.load(open(self.train_vector, 'rb'))
        train_label = []
        with open(self.train_label) as file:
            for line in file:
                train_label.append(int(line))
        # if sampling:
        #     _, train_vector, _, train_label = train_test_split(train_vector, train_label, test_size=sample,
        #                                                        random_state=seed)

        if sample:
            X_train, X_evl, y_train, y_evl = train_test_split(train_vector, train_label, test_size=0.5, random_state=seed)
            _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=2*sample, random_state=seed)
            _, X_evl, _, y_evl = train_test_split(X_evl, y_evl, test_size=2*sample, random_state=seed)


        print("Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_evl.shape[0]))
        start = time.time()
        print("Training Classifier...")
        self.classifier.fit(X_train, y_train)
        print("Training successfully in %s seconds " % (time.time() - start))
        # print("Intercept: ", self.classifier.intercept_)
        # print("Coefficient: ", self.classifier.coef_)
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
pipe.vectorize()
pipe.evaluate(sampling=True)
# pipe.classify()