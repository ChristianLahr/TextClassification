# Christian Lahr
# 07.07.2018

import pandas as pd
import numpy as np
import tqdm
tqdm.tqdm.pandas()

asset_path = r"AZD RNN/Assets"
train = pd.read_pickle(asset_path + "/train_data_firstPage.pickle")
eval = pd.read_pickle(asset_path + "/eval_data_firstPage.pickle")

def dataExploration(dataframe, explore_unigrams, explore_two_grams):

    print(dataframe.columns)
    print(dataframe.Text.head())
    print("Dataframe shape:", dataframe.shape)

    dataframe['word_count'] = dataframe['Text'].progress_apply(lambda x: len(str(x).split(" ")))
    dataframe['char_count'] = dataframe['Text'].progress_apply(lambda x: len(str(x)))

    print("Word and char counts:")
    print(dataframe[['Text','word_count', 'char_count']].head())
    print("Average number of words:", dataframe['word_count'].mean())
    print("Average text length:", dataframe['char_count'].mean())

    words = pd.Series(' '.join(dataframe['Text']).split()).value_counts()
    print("Most frequent words:\n", words.head())
    print("Number of unique words:", len(words))

    def find_ngrams(input_list, n):
        # list(zip(*[input_list[i:] for i in range(n)]))
        ngrams = []
        temp_list = input_list
        for i in range(len(input_list)):
            temp_list = input_list[i:]
            ngrams.append(temp_list[:n])
        return ngrams

    if explore_unigrams:
        dataframe["unigrams"] = dataframe['Text'].progress_apply(lambda x: find_ngrams(x, 1))
        unigrams = pd.Series([a for b in dataframe["unigrams"].tolist() for a in b]).value_counts()
        print("Most frequent unigrams:\n", unigrams.head())
        print("Number of unique chars incl. spaces:", len(unigrams))
        print("Chars:", unigrams.index)

    if explore_two_grams:
        dataframe["two_grams"] = dataframe['Text'].progress_apply(lambda x: find_ngrams(x, 2))
        two_grams = pd.Series([a for b in dataframe["two_grams"].tolist() for a in b]).value_counts()
        print("Most frequent two_grams:\n", two_grams.head())
        print("Number of unique two-grams:", len(two_grams))

train_explored = dataExploration(train, explore_unigrams=False, explore_two_grams=False)
eval_explored  = dataExploration(eval, explore_unigrams=True, explore_two_grams=False)

