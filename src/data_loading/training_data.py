import os
import warnings
import pandas as pd 
import numpy as np 
import pickle
from scipy.sparse import load_npz, vstack
from src.data_loading.interactions import METADATA_FILE, get_metadata
from src.preprocessing.vectorize_text import TFIDF_MATRIX
from src.utils import example_helper_function as warn 

warnings.warn = warn

ROOT = 'drive/MyDrive/CDS_Capstone_2022_Fall'
MBFC_LABELS_PATH = os.path.join(ROOT, 'data/Classification_Labels.csv')

def get_filtered_features(root=ROOT, additional_removed_words=[]):
    """
    Remove the names of political organizations or personalities from the TFIDF vectors 
    """
    # TF-IDF was trained using an old version 
    tfidf_file = os.path.join(root, 'models/balanced-dataset-tfidf-vectorizer.pickle')

    with open(tfidf_file, 'rb') as f:
        vectorizer = pickle.load(f)
    features = {v:k for k,v in vectorizer.vocabulary_.items()}

    political_entities = [
        'cnn',
        'democraci', 
        'trump', 
        'biden', 
        'breitbartnew', 
        'obama',
        'donald', 
        'foxnew', 
        'donaldjtrumpjr',
        'zerohedg', 
        'sentedcruz',
        'repmtg',
        'jsolomonreport',
        'gregabbotttx',
        'jimjordan',
        'timcast',
        'elonmusk',
        'hawleymo',
        "libsoftiktok",
        "alleg",
        "raheemkassam",
        "joerogan",
        "hawleymo",
        "laurenboebert",
        "tomfitton",
        "jennaellisesq",
        "mzhemingway",
        "dineshdsouza",
        "sebgorka",
        "georgepapa19",
        "randpaul",
        "reuter",
        "jackposobiec",
        "thebabylonbe",
        "mailonlin",
        "tomwint",
        "realdailywir",
        "repthomasmassi",
        "tpostmillenni",
        "repmattgaetz",
        "disclosetv",
        "thehil",
        "therealkeean",
        "mrandyngo",
        "phillewi",
        "ctvnew",
        "truenorthcentr",
        "repkinzing",
        "noblenatl",
        "melissarusso4ni",
        "njdotcom",
        "965tdi",
        "artistofthesumm",
        "laurenjauregui",
        "democrat", 
        "joe", 
        "prisonplanet",
        "hunter", 
        "pelosi", 
        "breaking911", 
        "nypost", 
        "fauci",
        "republican",
        "rsbnetwork",
        'nbcnew',
        "trudeau",
    ]
    misc_stop_words = [
        'none','video', 'also', 'year', 'im', 
        'said', 'need', 'one', 'much', 'via', 
        '1', '2', '3', '4', '5', 'day', 'look', 'shes', 'us',
        'e', 'dont', 'n', 'la', 'es', 'il', 'se', 'de', 'je',
    ]
    # remove miscellaneous stopwords and political entities
    filtered_features = [
        i for i, x in features.items()
        if x not in  
        political_entities + misc_stop_words + additional_removed_words
    ]
    return filtered_features


def load_twitter_features(root=ROOT):

    original_features = pd.read_csv(os.path.join(root,'data/url_feature_original_tweet.csv'))
    retweet_features = pd.read_csv(os.path.join(root,'data/url_feature_retweet.csv'))

    original_features.drop(columns=['url', 'domain', 'path', 'r', 'label'], inplace=True)
    retweet_features.drop(columns=['url', 'domain', 'path', 'r', 'label'], inplace=True)

    retweet_features.columns = [col +'_retweet' for col in retweet_features.columns]
    original_features.columns = [col +'_original' for col in original_features.columns]
    retweet_features.rename(columns={'id_hash256_retweet': 'id_hash256'}, inplace=True)
    original_features.rename(columns={'id_hash256_original': 'id_hash256'}, inplace=True)

    twitter_features = retweet_features.merge(original_features, on='id_hash256')

    return twitter_features


def load_data(root=ROOT):
    # collect metadata for URLS collected
    metadata = pd.read_json(os.path.join(root, 'data/capstone_url_metadata.json'))
    metadata_hc = pd.read_json(os.path.join(root, 'data/capstone_url_metadata_hc.json'))
    metadata_hc['domain'] = metadata_hc['url'].apply(
        lambda x: x.replace('https://www.', '').split('/', 1)[0]
    )

    # join with MBFC label 
    df_meta = pd.concat([metadata, metadata_hc])
    df_mbfc = pd.read_csv(os.path.join(root, 'data/mbfc.csv'))
    df_mbfc['label'] = df_mbfc['r'].map({"VH" : -1, "H" : -1,"MF" : 1, "M" : 1,"L" : 1,"VL" : 1})
    combined_metadata = pd.merge(df_meta, df_mbfc[['domain', 'label']], how='left', on='domain')
    
    # get TFIDF vector data 
    X = vstack([
        load_npz(os.path.join(root, 'data/balanced-dataset-tfidf-matrix-original.npz')), 
        load_npz(os.path.join(root, 'data/balanced-dataset-tfidf-matrix-hc.npz'))
    ])

    # get twitter data 
    twitter_data = load_twitter_features(root=root)
    twitter_data = combined_metadata[['id_hash256']].merge(twitter_data, on='id_hash256', how='left')
    return combined_metadata, X, twitter_data


def get_labeled_dataset(root=ROOT): 
    combined_metadata, X, twitter_data = load_data(root)
    labeled_metadata = combined_metadata[combined_metadata['label'].notna()]
    return labeled_metadata, X[labeled_metadata.index], twitter_data.iloc[labeled_metadata.index]


def get_unlabeled_dataset(): 
    combined_metadata, X, twitter_data = load_data()
    labeled_metadata = combined_metadata[combined_metadata['label'].isna()]
    return labeled_metadata, X[labeled_metadata.index], twitter_data.iloc[labeled_metadata.index]


def load_and_remove_politicial_entities(root=ROOT, dataset='combined', political_debias=True, additional_removed_words=[]):

    filtered_features = None
    combined_metadata, X, twitter_data = get_labeled_dataset(root) #balanced tfidf matrix 
    if political_debias:
        # remove key political entities names
        filtered_features = get_filtered_features(root, additional_removed_words)
        X = X[:, filtered_features]
    y = 1*(combined_metadata['label']==1)

    if dataset == 'combined':
        # combine twitter features with TF-IDF vectors 
        combined_data = pd.concat(
            [twitter_data.reset_index(drop=True), pd.DataFrame(X.toarray())], axis=1
        )
        combined_data.fillna(0, inplace=True)
        X = combined_data.drop(columns='id_hash256')
        X.columns = [str(x) for x in X.columns]
    
    elif dataset == 'twiitter':
        X = twitter_data
    elif dataset == 'tfidf':
        pass 
    else: 
        raise ValueError('Not implemented yet')
    
    return X, y, combined_metadata, filtered_features

