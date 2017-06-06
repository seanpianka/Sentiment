"""
sentiment.online_sentiment
~~~~~~~~~~~~~~~~~~~~~~~~~~

This module processes text through the sentiment analysis API

:author: Sean Pianka <sean.pianka@gmail.com>

.. seealso:: http://text-processing.com/docs/sentiment.html

"""
import json
import requests

import sys


class OnlineSentiment:
    def __init__(self):
        self._last_text = ''
        self._last_score = 0.0
        self._last_probabilities = {}

    def classify(self, text):
        return self.score_sentiment(self._analyze_text(text))

    def score_sentiment(self, raw_probabilities):
        probs = raw_probabilities

        score = 0. if probs['neutral'] > 0.5 else probs['pos'] - probs['neg']
        self._last_score = score

        return score

    def _classify_text(self, text):
        try:
            res = requests.post("https://japerk-text-processing.p.mashape.com/sentiment/",
                headers={
                    "X-Mashape-Key": "9Fp0ZKRkK5mshGbjZxPKQ1gHCqg0p1MzIcmjsnM6a17MnXlBwM",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json"
                },
                data={
                    "language": "english",
                    "text": text
                }
            ).json()
        except json.JSONDecodeError as e:
            print(e)
            self._last_probabilities = {
                'neutral': 1.00,
                'pos': 0.00,
                'neg': 0.00
            }
            self._last_text = text
            return self._last_probabilities
        except (KeyboardInterrupt, SystemExit) as e:
            raise e
        else:
            self._last_probabilities = res['probability']
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


if __name__ == '__main__':
    """
    usage outside of this file:

    from sentiment import Sentiment
    s = Sentiment()
    pos_neg_value = s.analyze("a string here")

    """
    s = Sentiment()

    while True:
        word = str(input('Enter a string: '))
        print("The value was {}".format(s.classify(word)))
