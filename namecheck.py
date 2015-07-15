#!/usr/bin/env python
from __future__ import print_function, division

import argparse
import urllib

data_urls = (
    "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/CSV_Database_of_First_Names.csv",
    "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/CSV_Database_of_Last_Names.csv",
    "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/chicago_employees.csv",
    "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/fifa_players_2012.csv",
    "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/olympicathletes.csv",
    "https://raw.githubusercontent.com/Avaaz/names_challenge/master/datasets/world_cup_players_en.js",
)

def download_datafiles(args):
    URLopener = urllib.URLopener()
    for url in data_urls:
        filename = url.split('/')[-1]
        URLopener.retrieve(url, filename)

def train_model(args):
    print(args)

def main(args):
    print(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=main)
    subparsers = parser.add_subparsers()

    parser_download = subparsers.add_parser("download", help="Download the training datasets")
    parser_download.set_defaults(func=download_datafiles)

    parser_train = subparsers.add_parser("train", help="Train the classifier")
    parser_train.set_defaults(func=train_model)
    
    args = parser.parse_args()
    args.func(args)
