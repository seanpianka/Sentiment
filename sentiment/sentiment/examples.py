"""
* tokenizing, separating by criteria
* lexicon and corporas
    corpora - body of text around similar topic (medical journals, presidential speeches)
    lexicon - words and their meanings
        * investor speak vs regular english speak
"""
import random
import pickle
import os

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union, gutenberg, wordnet, movie_reviews
from nltk.stem import WordNetLemmatizer
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC


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


def stopword_ex1():
    neg_text = "NNRQ has done well in the past few quarters, but is not expected to do well over the next year. TTWO was doing poorly, but is now on a steady increase that is expected to last for several quarters."
    filtered_sentence = [w for w in word_tokenize(neg_text) if w not in set(stopwords.words("english"))]
    print(filtered_sentence)


def process_content():
    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")
    custom_sent_tokenizer = PunktSentenceTokenizer(train_text)
    tokenized = custom_sent_tokenizer.tokenize(sample_text)

    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            namedEntity = nltk.ne_chunk(tagged)
            namedEntity.draw()
    except Exception as e:
        print(e)


def lem_ex1():
    lem = WordNetLemmatizer()
    print(lem.lemmatize("cacti"))


def corpus_ex1():
    sample = sent_tokenize(gutenberg.raw("bible-kjv.txt"))
    print(sample)


def wordnet_ex1():
    syns = wordnet.synsets("program")
    # synset
    print(syns[0].name())
    # just the word
    print(syns[0].lemmas()[0].name())
    # definition
    print(syns[0].definition())
    # examples
    print(syns[0].examples())


def wordnet_ex2():
    synonyms = []
    antonyms = []

    for syn in wordnet.synsets("good"):
        for l in syn.lemmas():
            print(l)
            synonyms.append(l.name())
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())

    print(set(synonyms))
    print(set(antonyms))


def wordnet_ex3():
    # semantic similarity
    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("boat.n.01")
    print(w1.wup_similarity(w2))
    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("dog.n.01")
    print(w1.wup_similarity(w2))
    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("car.n.01")
    print(w1.wup_similarity(w2))


def text_classifier():
    documents = [(list(movie_reviews.words(fileid)), category)
                 for category in movie_reviews.categories()
                 for fileid in movie_reviews.fileids(category)]

    random.shuffle(documents)
    stop_words = set(stopwords.words("english"))
    all_words = nltk.FreqDist([w.lower() for w in movie_reviews.words() if w not in stop_words])

    word_features = list(all_words.keys())[:3000]

    def find_features(document):
        words = set(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features

    print(find_features(movie_reviews.words('neg/cv000_29416.txt')))
    feature_sets = [(find_features(rev), category) for (rev, category) in documents]

    training_set = feature_sets[:1900]
    testing_set = feature_sets[1900:]

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

    print("Original NB Accuracy:", nltk.classify.accuracy(classifier, testing_set)*100)
    classifier.show_most_informative_features(15)
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

    print("MNB_NB Accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set)*100)
    ###########################################################################
    #GNB_classifier_pickle = "GNB_naivebayes.pkl"
    #if os.path.isfile(GNB_classifier_pickle):
    #    with open(GNB_classifier_pickle, "rb") as f:
    #        GNB_classifier = pickle.load(f)
    #else:
    #    # posterior = prior occurences * likelihood / evidence
    #    GNB_classifier = SklearnClassifier(GaussianNB())
    #    GNB_classifier.train(training_set)
    #    with open(GNB_classifier_pickle, "wb") as f:
    #        pickle.dump(GNB_classifier, f)

    #print("GNB_NB Accuracy:", nltk.classify.accuracy(GNB_classifier, testing_set)*100)
    #GNB_classifier.show_most_informative_features(15)
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

    print("BNB_NB Accuracy:", nltk.classify.accuracy(BNB_classifier, testing_set)*100)
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

    print("LOGREG_NB Accuracy:", nltk.classify.accuracy(LOGREG_classifier, testing_set)*100)
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

    print("SGD_NB Accuracy:", nltk.classify.accuracy(SGD_classifier, testing_set)*100)
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

    print("LSVC_NB Accuracy:", nltk.classify.accuracy(LSVC_classifier, testing_set)*100)
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

    print("NUSVC_NB Accuracy:", nltk.classify.accuracy(NUSVC_classifier, testing_set)*100)
    ###########################################################################

    voted_classifier = VoteClassifier(classifier,
                                      MNB_classifier,
                                      BNB_classifier,
                                      LOGREG_classifier,
                                      SGD_classifier,
                                      LSVC_classifier,
                                      NUSVC_classifier)
    print("voted_classifier Accuracy:", nltk.classify.accuracy(voted_classifier, testing_set)*100)
    print("Classification:", voted_classifier.classify(testing_set[0][0]),
          "Confidence %:", voted_classifier.confidence(testing_set[0][0]))
    print("Classification:", voted_classifier.classify(testing_set[1][0]),
          "Confidence %:", voted_classifier.confidence(testing_set[1][0]))
    print("Classification:", voted_classifier.classify(testing_set[2][0]),
          "Confidence %:", voted_classifier.confidence(testing_set[2][0]))
    print("Classification:", voted_classifier.classify(testing_set[3][0]),
          "Confidence %:", voted_classifier.confidence(testing_set[3][0]))
    print("Classification:", voted_classifier.classify(testing_set[4][0]),
          "Confidence %:", voted_classifier.confidence(testing_set[4][0]))


if __name__ == "__main__":
    text_classifier()

