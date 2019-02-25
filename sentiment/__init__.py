"""
* tokenizing, separating by criteria
* lexicon and corporas
    corpora - body of text around similar topic (medical journals, presidential speeches)
    lexicon - words and their meanings
        * investor speak vs regular english speak
"""
import json
import os
import pickle
import random
import requests
import sys
from statistics import mode

from nltk.classify import ClassifierI
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import stopwords, state_union, gutenberg, wordnet, movie_reviews
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC, NuSVC
import nltk


STOP_WORDS = set(stopwords.words("english"))
TRAINED_CLASSIFIERS_DIR = os.path.join(os.path.dirname(__file__), "trained")


class OnlineSentiment:
    """
    usage outside of this file:

    from sentiment import Sentiment
    s = Sentiment()
    pos_neg_value = s.analyze("a string here")

    s = Sentiment()

    while True:
        word = str(input('Enter a string: '))
        print("The value was {}".format(s.classify(word)))

    """

    def __init__(self):
        self._last_text = ""
        self._last_score = 0.0
        self._last_probabilities = {}

    def classify(self, text):
        return self.score_sentiment(self._analyze_text(text))

    def score_sentiment(self, raw_probabilities):
        probs = raw_probabilities

        score = 0. if probs["neutral"] > 0.5 else probs["pos"] - probs["neg"]
        self._last_score = score

        return score

    def _classify_text(self, text):
        try:
            res = requests.post(
                "https://japerk-text-processing.p.mashape.com/sentiment/",
                headers={
                    "X-Mashape-Key": "9Fp0ZKRkK5mshGbjZxPKQ1gHCqg0p1MzIcmjsnM6a17MnXlBwM",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"language": "english", "text": text},
            ).json()
        except json.JSONDecodeError as e:
            print(e)
            self._last_probabilities = {"neutral": 1.00, "pos": 0.00, "neg": 0.00}
            self._last_text = text
            return self._last_probabilities
        except (KeyboardInterrupt, SystemExit) as e:
            raise e
        else:
            self._last_probabilities = res["probability"]
            self._last_text = text
            return self._last_probabilities

    @property
    def last_score(self):
        return self._last_score

    @property
    def last_probabilities(self):
        return self._last_probabilities

    @property
    def last_text(self):
        return self._last_text


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        return mode([c.classify(features) for c in self._classifiers])

    def confidence(self, features):
        votes = [c.classify(features) for c in self._classifiers]
        choice_votes = votes.count(mode(votes))
        confidence_rating = choice_votes / len(votes)
        return confidence_rating


def text_classifier():
    with open("short_reviews/positive.txt", encoding="ISO-8859-1") as f:
        reviews = [(r, "pos") for r in f.read().splitlines()]
    with open("short_reviews/negative.txt", encoding="ISO-8859-1") as f:
        reviews += [(r, "neg") for r in f.read().splitlines()]

    all_words = nltk.FreqDist(
        [
            word
            for sublist in [word_tokenize(r[0]) for r in reviews]
            for word in sublist
            if word not in STOP_WORDS
        ]
    )
    word_features = list(all_words.keys())[:5000]

    def find_features(document):
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = w in words
        return features

    feature_sets = [(find_features(rev), category) for (rev, category) in reviews]
    random.shuffle(feature_sets)  # mix neg/pos reviews

    training_set = feature_sets[:10000]  # 10600
    testing_set = feature_sets[10000:]

    ###########################################################################
    classifier_pickle = "naivebayes.pkl"
    if os.path.isfile(classifier_pickle):
        with open(classifier_pickle, "rb") as f:
            classifier = pickle.load(f)
    else:
        # posterior = prior occurences * likelihood / evidence
        classifier = nltk.NaiveBayesClassifier.train(training_set)
        with open(classifier_pickle, "wb") as f:
            pickle.dump(classifier, f)

    print(
        "Original NB Accuracy:", nltk.classify.accuracy(classifier, testing_set) * 100
    )
    # classifier.show_most_informative_features(15)
    ###########################################################################
    MNB_classifier_pickle = "MNB_naivebayes.pkl"
    if os.path.isfile(MNB_classifier_pickle):
        with open(MNB_classifier_pickle, "rb") as f:
            MNB_classifier = pickle.load(f)
    else:
        # posterior = prior occurences * likelihood / evidence
        MNB_classifier = SklearnClassifier(MultinomialNB())
        MNB_classifier.train(training_set)
        with open(MNB_classifier_pickle, "wb") as f:
            pickle.dump(MNB_classifier, f)

    print("MNB_NB Accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set) * 100)
    ###########################################################################
    BNB_classifier_pickle = "BNB_naivebayes.pkl"
    if os.path.isfile(BNB_classifier_pickle):
        with open(BNB_classifier_pickle, "rb") as f:
            BNB_classifier = pickle.load(f)
    else:
        # posterior = prior occurences * likelihood / evidence
        BNB_classifier = SklearnClassifier(BernoulliNB())
        BNB_classifier.train(training_set)
        with open(BNB_classifier_pickle, "wb") as f:
            pickle.dump(BNB_classifier, f)

    print("BNB_NB Accuracy:", nltk.classify.accuracy(BNB_classifier, testing_set) * 100)
    ###########################################################################
    LOGREG_classifier_pickle = "LOGREG_naivebayes.pkl"
    if os.path.isfile(LOGREG_classifier_pickle):
        with open(LOGREG_classifier_pickle, "rb") as f:
            LOGREG_classifier = pickle.load(f)
    else:
        # posterior = prior occurences * likelihood / evidence
        LOGREG_classifier = SklearnClassifier(LogisticRegression())
        LOGREG_classifier.train(training_set)
        with open(LOGREG_classifier_pickle, "wb") as f:
            pickle.dump(LOGREG_classifier, f)

    print(
        "LOGREG_NB Accuracy:",
        nltk.classify.accuracy(LOGREG_classifier, testing_set) * 100,
    )
    ###########################################################################
    SGD_classifier_pickle = "SGD_naivebayes.pkl"
    if os.path.isfile(SGD_classifier_pickle):
        with open(SGD_classifier_pickle, "rb") as f:
            SGD_classifier = pickle.load(f)
    else:
        # posterior = prior occurences * likelihood / evidence
        SGD_classifier = SklearnClassifier(SGDClassifier())
        SGD_classifier.train(training_set)
        with open(SGD_classifier_pickle, "wb") as f:
            pickle.dump(SGD_classifier, f)

    print("SGD_NB Accuracy:", nltk.classify.accuracy(SGD_classifier, testing_set) * 100)
    ###########################################################################
    LSVC_classifier_pickle = "LSVC_naivebayes.pkl"
    if os.path.isfile(LSVC_classifier_pickle):
        with open(LSVC_classifier_pickle, "rb") as f:
            LSVC_classifier = pickle.load(f)
    else:
        # posterior = prior occurences * likelihood / evidence
        LSVC_classifier = SklearnClassifier(LinearSVC())
        LSVC_classifier.train(training_set)
        with open(LSVC_classifier_pickle, "wb") as f:
            pickle.dump(LSVC_classifier, f)

    print(
        "LSVC_NB Accuracy:", nltk.classify.accuracy(LSVC_classifier, testing_set) * 100
    )
    ###########################################################################
    NUSVC_classifier_pickle = "NUSVC_naivebayes.pkl"
    if os.path.isfile(NUSVC_classifier_pickle):
        with open(NUSVC_classifier_pickle, "rb") as f:
            NUSVC_classifier = pickle.load(f)
    else:
        # posterior = prior occurences * likelihood / evidence
        NUSVC_classifier = SklearnClassifier(NuSVC())
        NUSVC_classifier.train(training_set)
        with open(NUSVC_classifier_pickle, "wb") as f:
            pickle.dump(NUSVC_classifier, f)

    print(
        "NUSVC_NB Accuracy:",
        nltk.classify.accuracy(NUSVC_classifier, testing_set) * 100,
    )
    ###########################################################################

    voted_classifier = VoteClassifier(
        classifier,
        MNB_classifier,
        BNB_classifier,
        LOGREG_classifier,
        SGD_classifier,
        LSVC_classifier,
        NUSVC_classifier,
    )
    print(
        "voted_classifier Accuracy:",
        nltk.classify.accuracy(voted_classifier, testing_set) * 100,
    )
    print(
        "Classification:",
        voted_classifier.classify(testing_set[0][0]),
        "Confidence %:",
        voted_classifier.confidence(testing_set[0][0]),
    )
    print(
        "Classification:",
        voted_classifier.classify(testing_set[1][0]),
        "Confidence %:",
        voted_classifier.confidence(testing_set[1][0]),
    )
    print(
        "Classification:",
        voted_classifier.classify(testing_set[2][0]),
        "Confidence %:",
        voted_classifier.confidence(testing_set[2][0]),
    )
    print(
        "Classification:",
        voted_classifier.classify(testing_set[3][0]),
        "Confidence %:",
        voted_classifier.confidence(testing_set[3][0]),
    )
    print(
        "Classification:",
        voted_classifier.classify(testing_set[4][0]),
        "Confidence %:",
        voted_classifier.confidence(testing_set[4][0]),
    )


if __name__ == "__main__":
    text_classifier()
