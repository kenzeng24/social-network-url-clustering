import pandas as pd 
import numpy as np 
import json, re, os
import spacy, string, nltk
from tqdm.notebook import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import preprocessing
from nltk import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

translator=str.maketrans(string.punctuation, ' ' * len(string.punctuation))
string.punctuation
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

ROOT = 'drive/MyDrive/CDS_Capstone_2022_Fall'
METADATA_FILE = os.path.join(ROOT, 'data/capstone_url_metadata.json')
AGGREGATE_DATA_PATH = os.path.join(ROOT, 'data/data')
EMBEDDING_PATH = os.path.join(ROOT, 'embedding')
CLUSTER_FILE = os.path.join(ROOT, 'data/cluster_generated_reduced.json')

def clean_text(text):
    '''remove urls from text'''
    return re.sub('http\S+', '', text) 


def retrieve_interactions(id):
    '''return all '''
    json_filename = "{}/{}/{}.json".format(AGGREGATE_DATA_PATH, id[:3], id)  
    result = json.load(open(json_filename, 'r'))
    return result


def aggregate_interaction_text(id, i_min=1000):
    """TODO: add additional filters"""
    result = retrieve_interactions(id)
    output_text = ''
    for platform_data in result['result'].values():
        for data in platform_data:
            if data['i'] > i_min:
                output_text += clean_text(data['d'].lower()) + " "
    output = output_text[:-1]
    if not len(output):
        output = '<empty>'
    return output # remove empty space at the end


def aggregate_text(metadata, **kwargs):
    tqdm.pandas()
    return metadata.id_hash256.progress_apply(
        lambda id: aggregate_interaction_text(id, **kwargs))


def get_metadata(filename=METADATA_FILE):
    return pd.read_json(filename)

    



