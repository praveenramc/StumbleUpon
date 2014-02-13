from utility import *
from data import get_labels

from sklearn.metrics import roc_auc_score
from scipy.optimize import nnls
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import KFold


def calculateRocScore(s):
    labels = get_labels()
    v = roc_auc_score(labels, s[:len(labels)])
    return round(v, 5)


def getWeightforLabel(data, labels):
    weights, _ = nnls(data[:len(labels)], labels)
    return data.dot(weights)

def getSelectedWeight(data, labels):
    weights, _ = nnls(data[:len(labels)], labels)
    return data[:, weights > 0].mean(axis=1)


def main():
    scores = load('scores')
    labels = get_labels()

    scores_with_raw = np.vstack(scores.values()).T
    scores_without_raw = np.vstack([scores[n] for n in scores if ('raw:' not in n)]).T

    print 'Best Model:',
    print max([(calculateRocScore(scores[name]), name) for name in scores])
    print
    print calculateRocScore(scores_with_raw.mean(axis=1)),
    print 'Average data with raw'
    print calculateRocScore(getWeightforLabel(scores_with_raw, labels)),
    print 'Weighted with raw data'
    print calculateRocScore(getSelectedWeight(scores_with_raw, labels)),
    print 'Selected Weight with raw'
    print
    print calculateRocScore(scores_without_raw.mean(axis=1)),
    print 'Mean without raw data'
    print calculateRocScore(getWeightforLabel(scores_without_raw, labels)),
    print 'Weighted data without raw'
    print calculateRocScore(getSelectedWeight(scores_without_raw, labels)),
    print 'Selected weight without raw data'
    print

    final = getSelectedWeight(scores_without_raw, labels)
    submit(final[len(labels):])


if __name__ == "__main__":
    main()
