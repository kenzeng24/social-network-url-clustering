import os 
import pandas as pd 
import numpy as np 
from src.data_loading.interactions import ROOT


def load_twitter_features():

    original_features = pd.read_csv(os.path.join(ROOT,'data/url_feature_original_tweet.csv'))
    retweet_features = pd.read_csv(os.path.join(ROOT,'data/url_feature_retweet.csv'))

    unlabeled = original_features.label.isin(['?'])
    original_features = original_features.loc[~unlabeled]
    retweet_features = retweet_features.loc[~unlabeled]

    assert np.mean(original_features.label == retweet_features.label)
    labels = 1* (original_features.label.isin('0.0', '1.0'))

    original_features.drop(columns=['url', 'domain', 'path', 'r', 'label'], inplace=True)
    retweet_features.drop(columns=['url', 'domain', 'path', 'r', 'label'], inplace=True)

    retweet_features.columns = [col +'_retweet' for col in retweet_features.columns]
    original_features.columns = [col +'_original' for col in original_features.columns]

    twitter_features = retweet_features.merge(
        original_features, 
        left_on='id_hash256_retweet', 
        right_on='id_hash256_original'
    ).drop(columns=[
        'id_hash256_retweet', 
        'id_hash256_original'])

    return twitter_features, labels