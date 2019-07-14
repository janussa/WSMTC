import re
import xml.sax.saxutils as saxutils

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from pandas import DataFrame

import numpy as np

np.random.seed(42)
data_folder = 'reuters21578/'

sgml_number_of_files = 22
sgml_file_name_template = 'reut2-{}.sgm'
category_files = {
    'to_': ('Topics', 'all-topics-strings.lc.txt'),
    'pl_': ('Places', 'all-places-strings.lc.txt'),
    'pe_': ('People', 'all-people-strings.lc.txt'),
    'or_': ('Organizations', 'all-orgs-strings.lc.txt'),
    'ex_': ('Exchanges', 'all-exchanges-strings.lc.txt')
}

category_data = []

for category_prefix in category_files.keys():
    with open(data_folder + category_files[category_prefix][1], 'r') as file:
        for category in file.readlines():
            category_data.append([category_prefix + category.strip().lower(),
                                  category_files[category_prefix][0],
                                  0])

news_categories = DataFrame(data=category_data, columns=['Name', 'Type', 'Newslines'])


def update_frequencies(categories):
    for c in categories:
        idx = news_categories[news_categories.Name == c].index[0]
        f = news_categories.get_value(idx, 'Newslines')
        news_categories.set_value(idx, 'Newslines', f + 1)


def to_category_vector(categories, target_categories):
    vector = np.zeros(len(target_categories)).astype(np.float32)

    for t in range(len(target_categories)):
        if target_categories[t] in categories:
            vector[t] = 1.0

    return vector


selected_categories = ['pl_usa', 'to_earn', 'to_acq', 'pl_uk', 'pl_japan', 'pl_canada', 'to_money-fx',
                       'to_crude', 'to_grain', 'pl_west-germany', 'to_trade', 'to_interest',
                       'pl_france', 'or_ec', 'pl_brazil', 'to_wheat', 'to_ship', 'pl_australia',
                       'to_corn', 'pl_china']

# Parse SGML files
document_X = []
document_Y = []


def strip_tags(text):
    return re.sub('<[^<]+?>', '', text).strip()


def unescape(text):
    return saxutils.unescape(text)


# Iterate all files
for i in range(sgml_number_of_files):
    file_name = sgml_file_name_template.format(str(i).zfill(3))
    print('Reading file: %s' % file_name)

    with open(data_folder + file_name, 'rb') as file:
        content = BeautifulSoup(file.read().lower(), "lxml")

        for newsline in content('reuters_20'):
            document_categories = []

            # News-line Id
            document_id = newsline['newid']

            # News-line text
            document_body = strip_tags(str(newsline('text')[0].text)).replace('reuter\n&#3;', '')
            document_body = unescape(document_body)

            # News-line categories
            topics = newsline.topics.contents
            places = newsline.places.contents
            people = newsline.people.contents
            orgs = newsline.orgs.contents
            exchanges = newsline.exchanges.contents

            for topic in topics:
                document_categories.append('to_' + strip_tags(str(topic)))

            for place in places:
                document_categories.append('pl_' + strip_tags(str(place)))

            for person in people:
                document_categories.append('pe_' + strip_tags(str(person)))

            for org in orgs:
                document_categories.append('or_' + strip_tags(str(org)))

            for exchange in exchanges:
                document_categories.append('ex_' + strip_tags(str(exchange)))

            # Create new document
            update_frequencies(document_categories)

            document_X.append(document_body)
            document_Y.append(to_category_vector(document_categories, selected_categories))
lemmatizer = WordNetLemmatizer()
strip_special_chars = re.compile("[^A-Za-z0-9,. ]+")
stop_words = set(stopwords.words("english"))


def clean_up_sentence(r, sw=None):
    r = r.lower().replace("<br />", " ")
    r = re.sub(strip_special_chars, "", r.lower())
    if sw is not None:
        words = word_tokenize(r)
        filtered_sentence = []
        for w in words:
            w = lemmatizer.lemmatize(w)
            if w not in sw:
                filtered_sentence.append(w)
        return " ".join(filtered_sentence)
    else:
        return r


totalX = []
totalY = np.array(document_Y)
for i, doc in enumerate(document_X):
    totalX.append(clean_up_sentence(doc, stop_words))
