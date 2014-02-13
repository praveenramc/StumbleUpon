import json, re

from utility import *
from collections import Counter
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
from parse_html import clean_string, TAGS


def main():
    # load data
    path = 'exracted_content/extractedrawtext'
    data = map(json.loads, file(path))

    # count word for every tag
    tags = TAGS + ['boilerplate', 'boilerpipe']
    counts_per_tag = {}

    for eachtag in tags:
        counts = map(counter, getItems(eachtag, data))
        counts_per_tag[eachtag] = counts

    total = sumUp(counts_per_tag, len(data))

    # vectorize
    vect = DictVectorizer()
    vect.fit([total])

    features = {}
    for eachtag in tags:
        features[eachtag] = vect.transform(counts_per_tag[eachtag])

    save('textfeature', features)
    save('textvector', vect)


def createTokens(string):
    string = re.sub(r'[0-9]', '0', string)
    words = re.split(r'\W+', string)

    return map(None, words)


def counter(texts):
    looper = Counter()

    for text in texts:
        words = createTokens(text)
        looper.update(words)

    return looper


def getItems(tag, items):
    for item in items:
        yield item[tag] if (tag in item) else []

    total = Counter()


def sumUp(counts, n):
    tags = list(counts)
    total = Counter()

    for i in xrange(n):
        words = set()
        for tag in tags:
            words.update(set(counts[tag][i]))

        total.update(words)

    return total

if __name__ == "__main__":
    main()
