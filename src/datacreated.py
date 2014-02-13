import csv, json, re
from collections import defaultdict, Counter

from unidecode import unidecode
from utility import *


def extractDomainForAURL(url):
    # extract domains
    domain = url.lower().split('/')[2]
    domain_parts = domain.split('.')

    # e.g. co.uk
    if domain_parts[-2] not in ['com', 'co']:
        return '.'.join(domain_parts[-2:])
    else:
        return '.'.join(domain_parts[-3:])


def loadTheData(filename):
    csv_file_object = csv.reader(file(filename, 'rb'), delimiter='\t')

    header = csv_file_object.next()

    data = []

    for eachrow in csv_file_object:
        # make dictionary
        item = {}
        for i in range(len(header)):
            item[header[i]] = eachrow[i]

        # url
        item['real_url'] = item['url'].lower()
        item['domain'] = extractDomainForAURL(item['url'])
        item['tld'] = item['domain'].split('.')[-1]

        # parse boilerplate
        boilerplate = json.loads(item['boilerplate'])
        for f in ['title', 'url', 'body']:
            item[f] = boilerplate[f] if (f in boilerplate) else u''
            item[f] = unidecode(item[f]) if item[f] else ''

        del item['boilerplate']

        # label
        if 'label' in item:
            item['label'] = item['label'] == '1'
        else:
            item['label'] = '?'

        data.append(item)

    return data


def getTheTrainData():
    return load('train', lambda: loadTheData('data/train.tsv'))


def getTheTestData():
    return load('test', lambda: loadTheData('data/test.tsv'))


def getTheLabels():
    return np.array([item['label'] for item in getTheTrainData()])
