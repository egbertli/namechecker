#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import cPickle
import random
import string
import urllib

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


data_urls = {
    "first_names": "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/CSV_Database_of_First_Names.csv",
    "last_names": "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/CSV_Database_of_Last_Names.csv",
    "chicago": "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/chicago_employees.csv",
    "fifa": "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/fifa_players_2012.csv",
    "olympics": "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/olympicathletes.csv",
    "dictionary": "http://svnweb.freebsd.org/csrg/share/dict/words?view=co",
    # "world_cup": "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/world_cup_players_en.js",
}


def download_datafiles(args):
    '''Downloads data files to the current directory'''
    URLopener = urllib.URLopener()
    for url in data_urls.itervalues():
        filename = url.split('?')[0].split('/')[-1]
        URLopener.retrieve(url, filename)


def load_datafiles():
    '''Load the datafiles into pandas'''
    data = {}
    for name, url in data_urls.iteritems():
        filename = url.split('?')[0].split('/')[-1]
        try:
            data[name] = pd.read_csv(filename)
        except:
            continue
    return data


def clean_data(data):
    '''Do some data cleaning'''
    cleaned_data = {}
    cleaned_data['first_names'] = data['first_names'].iloc[:,0].str.decode('utf-8')
    cleaned_data['last_names'] = data['last_names'].iloc[:,0].str.decode('utf-8')
    cleaned_data['chicago'] = data['chicago']['Name'].str.decode('utf-8')
    cleaned_data['fifa'] = data['fifa'][' name'].str.decode('utf-8')
    cleaned_data['olympics'] = data['olympics']['Athlete'].str.decode('latin1')
    cleaned_data['dictionary'] = data['dictionary'].iloc[:,0].str.decode('utf-8')

    for item, series in cleaned_data.iteritems():
        series = series.str.strip()

        if item == 'chicago':
            # Change from "Last, First M" to "First M Last"
            series = series.str.split(',\W*').str[::-1].str.join(' ')
        if item == 'olympics':
            # Filter duplicates
            series = pd.Series(series.unique())
            series = series[~series.isnull()]
        if item == 'dictionary':
            # Remove proper names
            series = series[~series.str.istitle()]
        
        # Add start and end of name delimiters
        if item == 'first_names':
            series = '{' + series
        elif item == 'last_names':
            series = series + '}'
        elif item == 'dictionary':
            pass
        else:
            series = '{' + series + '}' 
        
        cleaned_data[item] = series

    return cleaned_data


def train_model(args):
    '''Train a Naive Bayes n-gram character classifier'''
    data_files = load_datafiles()
    cleaned_data = clean_data(data_files)

    # Name data
    data = {'text': [], 'class': []}
    for item in ('first_names', 'last_names', 'chicago', 'fifa', 'olympics'):
        name_data = cleaned_data[item]
        for name in name_data:
            data['text'].append(name)
            data['class'].append('name')

    # Feed in some random strings for negative examples
    n = len(data['text'])
    characters = unicode(string.printable)
    current_names = list(data['text'])
    for name in current_names:
        gibberish = ''.join(random.sample(characters, len(name)))
        data['text'].append(gibberish)
        data['class'].append('not_name')
    
    # Feed in some fake names composed of random non-name English words
    for item in ('dictionary',):
        word_data = cleaned_data[item]
        for ii in xrange(int(n/4)):
            # Two word 'names'
            rows = np.random.choice(word_data.values, 2)
            fake_name = ' '.join(rows)
            # Start and end-of-name delimiters
            data['text'].append('{' + fake_name + '}')
            data['class'].append('not_name')
            
            # Three word 'names'
            rows = np.random.choice(word_data.values, 3)
            fake_name = ' '.join(rows)
            # Start and end-of-name delimiters
            data['text'].append('{' + fake_name + '}')
            data['class'].append('not_name')
            
            # Four word 'names'
            rows = np.random.choice(word_data.values, 4)
            fake_name = ' '.join(rows)
            # Start and end-of-name delimiters
            data['text'].append('{' + fake_name + '}')
            data['class'].append('not_name')

    data = pd.DataFrame(data)

    # Tokenize the exmples by breaking them up into 2- and 3-character substrings
    cv = CountVectorizer(
        analyzer='char',
        strip_accents='unicode',
        ngram_range=(2,3),
        lowercase=True,
    )
    # Feed everything into a Multinomial Naive Bayes classifier
    pipeline = Pipeline([
        ('count_vectorizer', cv),
        ('classifier', MultinomialNB())
    ])

    train_text = data['text'].values
    train_y = data['class'].values.astype(str)
    pipeline.fit(train_text, train_y)

    cPickle.dump(pipeline, open('name_model.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)


class NameValidator(object):
    def __init__(self):
        self.pipeline = cPickle.load(open('name_model.pkl', 'rb'))
        
    def simple_checks(self, name):
        valid = True
    
        # Check for empty name
        valid &= name != ""
    
        # Check for more than 5 words (First Middle1 Middle2 Last [Jr/Sr/III])
        valid &= len(name.split()) <= 5

        # Check for no numbers
        valid &= not any(char.isdigit() for char in name)

        # Check for no punctuation
        punctuation = '''!#$%&*+/:;<=>@[\]^_`{|}~"'''
        valid &= not any(char in punctuation for char in name)

        return valid

    def machine_learning_checks(self, name):
        name = unicode('{' + name + '}')
        prediction = self.pipeline.predict([name])
        return prediction[0] == 'name'

    def validate_name(self, full_name):
        name = full_name
        name.strip()

        valid = True
        valid &= self.simple_checks(full_name)
        valid &= self.machine_learning_checks(full_name)
        return valid


def main(args):
    name_validator = NameValidator()
    print(name_validator.validate_name(args.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
    """
    Validate whether an entered string represents a Romanized name.
    On first run requires you to first download leaning data with the 'download' command
    then train a model with the 'train' command.  After the model is trained the 'validate'
    command can be run on an input name."""
    )
    subparsers = parser.add_subparsers()

    parser_download = subparsers.add_parser("download", help="Download the training datasets")
    parser_download.set_defaults(func=download_datafiles)

    parser_train = subparsers.add_parser("train", help="Train the classifier")
    parser_train.set_defaults(func=train_model)
    
    parser_validate = subparsers.add_parser("validate", help="Validate a name")
    parser_validate.set_defaults(func=main)
    parser_validate.add_argument('name', help="Name to Validate")
    
    args = parser.parse_args()
    args.func(args)
