import os
import pandas as pd 
import numpy as np 
from scipy.sparse import load_npz, vstack
from src.data_loading.interactions import ROOT, METADATA_FILE, get_metadata
from src.preprocessing.vectorize_text import TFIDF_MATRIX
from sklearn.model_selection import train_test_split

MBFC_LABELS_PATH = os.path.join(ROOT, 'data/Classification_Labels.csv')


def get_tfidf_dataset(
    metadata_file=METADATA_FILE,
    tfidf_matrix_file=TFIDF_MATRIX, 
    labels_file=MBFC_LABELS_PATH):
    """
    get TFIDF vectors and MBFC labels of URLs in the MBFC dataset
    """
    MBFC_set=pd.read_csv(labels_file)
    tfidf_sparse=load_npz(tfidf_matrix_file)
    metadata = pd.read_json(metadata_file)

    filtered_nonnull = MBFC_set[MBFC_set['NaNs_Per_Row'] ==0]

    # keep 
    X = tfidf_sparse[filtered_nonnull.index,:]
    y = filtered_nonnull['Credibility_Rank'] # for now 
    urls = metadata[['url', 'id_hash256']].iloc[filtered_nonnull.index]
    return X,y,urls


def get_balanced_tfidf_data():

    # get the tfidf vectors from original dataset 
    tfidf_matrix_file =  os.path.join(ROOT, 'data/balanced-dataset-tfidf-matrix-original.npz')
    X, y, urls = get_tfidf_dataset(tfidf_matrix_file=tfidf_matrix_file) 

    # get metadata and tfidf matrix from new dataset 
    metadata_hc = get_metadata(filename=os.path.join(ROOT, 'data/capstone_url_metadata_hc.json'))
    X_hc = load_npz(os.path.join(ROOT, 'data/balanced-dataset-tfidf-matrix-hc.npz'))
    y_hc = np.ones(shape=X_hc.shape[0]) # all urls in this sample are not suspiciious
    urls_hc = metadata_hc[['url', 'id_hash256']]

    # combine the outputs from both data sources 
    X_combined = vstack([X, X_hc])
    y_combined = np.hstack([y.values, y_hc])
    urls_combined =  pd.concat([urls, urls_hc])
    return X_combined, y_combined, urls_combined


def get_unabeled_urls(
    metadata_file=METADATA_FILE,
    tfidf_matrix_file=TFIDF_MATRIX, 
    labels_file=MBFC_LABELS_PATH):
    """
    get TFIDF vectors of URLs without an MBFC label 
    """
    MBFC_set=pd.read_csv(labels_file)
    tfidf_sparse=load_npz(tfidf_matrix_file)
    metadata = pd.read_json(metadata_file)

    unlabeled = MBFC_set[MBFC_set['NaNs_Per_Row'] !=0]
    X = tfidf_sparse[unlabeled.index,:]
    urls = metadata[['url', 'id_hash256']].iloc[unlabeled.index]
    return X,urls


def load_twitter_features():

    original_features = pd.read_csv(os.path.join(ROOT,'data/url_feature_original_tweet.csv'))
    retweet_features = pd.read_csv(os.path.join(ROOT,'data/url_feature_retweet.csv'))

    original_features.drop(columns=['url', 'domain', 'path', 'r', 'label'], inplace=True)
    retweet_features.drop(columns=['url', 'domain', 'path', 'r', 'label'], inplace=True)

    retweet_features.columns = [col +'_retweet' for col in retweet_features.columns]
    original_features.columns = [col +'_original' for col in original_features.columns]
    retweet_features.rename(columns={'id_hash256_retweet': 'id_hash256'}, inplace=True)
    original_features.rename(columns={'id_hash256_original': 'id_hash256'}, inplace=True)

    twitter_features = retweet_features.merge(original_features, on='id_hash256')

    return twitter_features


def load_data():
    # collect metadata for URLS collected
    metadata = pd.read_json(os.path.join(ROOT, 'data/capstone_url_metadata.json'))
    metadata_hc = pd.read_json(os.path.join(ROOT, 'data/capstone_url_metadata_hc.json'))
    metadata_hc['domain'] = metadata_hc['url'].apply(
        lambda x: x.replace('https://www.', '').split('/', 1)[0]
    )

    # join with MBFC label 
    df_meta = pd.concat([metadata, metadata_hc])
    df_mbfc = pd.read_csv(os.path.join(ROOT, 'data/mbfc.csv'))
    df_mbfc['label'] = df_mbfc['r'].map({"VH" : -1, "H" : -1,"MF" : 1, "M" : 1,"L" : 1,"VL" : 1})
    combined_metadata = pd.merge(df_meta, df_mbfc[['domain', 'label']], how='left', on='domain')
    
    # get TFIDF vector data 
    X = vstack([
        load_npz(os.path.join(ROOT, 'data/balanced-dataset-tfidf-matrix-original.npz')), 
        load_npz(os.path.join(ROOT, 'data/balanced-dataset-tfidf-matrix-hc.npz'))
    ])

    # get twitter data 
    twitter_data = load_twitter_features()
    twitter_data = combined_metadata[['id_hash256']].merge(twitter_data, on='id_hash256', how='left')
    return combined_metadata, X, twitter_data


def get_labeled_dataset(): 
    combined_metadata, X, twitter_data = load_data()
    labeled_metadata = combined_metadata[combined_metadata['label'].notna()]
    return labeled_metadata, X[labeled_metadata.index], twitter_data.iloc[labeled_metadata.index]

def get_unlabeled_dataset(): 
    combined_metadata, X, twitter_data = load_data()
    labeled_metadata = combined_metadata[combined_metadata['label'].isna()]
    return labeled_metadata, X[labeled_metadata.index], twitter_data.iloc[labeled_metadata.index]

