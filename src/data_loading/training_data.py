import os
import pandas as pd 
import numpy as np 
from scipy.sparse import load_npz, vstack
from src.data_loading.interactions import ROOT, METADATA_FILE, get_metadata
from src.preprocessing.vectorize_text import TFIDF_MATRIX

MBFC_LABELS_PATH = os.path.join(ROOT, 'data/Classification_Labels.csv')


def get_tfidf_dataset(
    metadata_file=METADATA_FILE,
    tfidf_matrix_file=TFIDF_MATRIX, 
    labels_file=MBFC_LABELS_PATH):
    """
    get TFIDF vectors and MBFC labels of URLs in the MBFC dataset
    """
    MBFC_set=pd.read_csv(labels_file)
    tfidf_sparse=sparse.load_npz(tfidf_matrix_file)
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
    tfidf_sparse=sparse.load_npz(tfidf_matrix_file)
    metadata = pd.read_json(metadata_file)

    unlabeled = MBFC_set[MBFC_set['NaNs_Per_Row'] !=0]
    X = tfidf_sparse[unlabeled.index,:]
    urls = metadata[['url', 'id_hash256']].iloc[unlabeled.index]
    return X,urls