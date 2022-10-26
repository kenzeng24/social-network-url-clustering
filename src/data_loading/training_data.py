import os
import pandas as pd 
import scipy.sparse as sparse

from src.data_loading.interactions import ROOT, METADATA_FILE
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