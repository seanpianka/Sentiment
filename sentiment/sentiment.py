"""
sentiment
~~~~~~~~~

This module processes text through the sentiment analysis API

:author: Sean Pianka <sean.pianka@gmail.com>

.. seealso:: http://text-processing.com/docs/sentiment.html

"""
import requests

SENTIMENT_API_URL = 'http://text-processing.com/api/sentiment/'

class Sentiment:
    def __init__(self, api_url=SENTIMENT_API_URL):
        self._api_url = api_url
        self._last_text = ''
        self._last_score = 0.0
        self._last_probabilities = {}

    def analyze(self, text):
        return self.score_sentiment(self._analyze_text(text, self._api_url))

    def score_sentiment(self, raw_probabilities):
        probs = raw_probabilities

        score = 0. if probs['neutral'] > 0.5 else probs['pos'] - probs['neg']
        self._last_score = score

        return score

    def _analyze_text(self, text, api_url=None):
        """Sentiment analysis

        :returns:
            {
                "label": "pos", "neg" or "neutral",
                "probability": {
                    "neg": float,
                    "neutral": float,
                    "pos": float
                }
            }

        :raises:
            - requests.exceptions.RequestException
            - requests.exceptions.HTTPError
            - json.JSONDecodeError

        """
        if not api_url:
            raise RuntimeError('No language processing API URL provided.')

        response = requests.post(SENTIMENT_API_URL, data={'text': text})
        response.raise_for_status()

        self._last_text = text
        self._last_probabilities = response.json()['probability']

        return self._last_probabilities

    @property
    def api_url(self):
        return self.api_url

    @api_url.setter
    def api_url(self, new_api_url):
        if not new_api_url or not isinstance(new_api_url, str):
            pass
        else:
            self._api_url = new_api_url

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
        print("The value was {}".format(s.analyze(word)))
