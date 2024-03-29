import pandas as pd 
import numpy as np 
import json, re, os
import string, nltk
from tqdm import tqdm

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import preprocessing
from nltk import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords',  quiet=True)

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

    # remove URLs 
    clean_text = re.sub('http\S+', '', text) 
    # remove non-alphanumeric characters 
    clean_text = re.compile('[^a-zA-Z\d\s]+').sub('', clean_text)
    return clean_text


def retrieve_interactions(id):
    '''return all '''
    json_filename = "{}/{}/{}.json".format(AGGREGATE_DATA_PATH, id[:3], id)  
    result = json.load(open(json_filename, 'r'))
    return result


def aggregate_interaction_text(id, i_min=1000, first_n=1000000):
    """
    combine and clean the interaction text for a urlid 
        only analyzing the first_n interaction texts from each domain 
        with more than i_min interactions 
    """
    result = retrieve_interactions(id)
    output_text = ''
    for platform, platform_data in result['result'].items():
        if platform != 'retweet':
            count = 0 
            for data in platform_data:
                # use all interactions if number of interactions less than min
                if len(platform_data) <= first_n or data['i'] > i_min:
                    output_text += clean_text(data['d'].lower()) + " "
                    count += 1 
                # stop after the first_n posts with high interactions
                if count > first_n:
                    break
    # ignore the empty space at the end
    output = output_text[:-1]

    if not len(output):
        output = '<empty>'
    return output # remove empty space at the end


def aggregate_text(url_ids, filename, **kwargs):
    """
    aggregate the interaction text for each urlid 
    and save results as a csv file under filename
    """
    if os.path.exists(filename):
        raise ValueError(f'file {filename} already exists. use a different filename')

    agg_text_df = pd.DataFrame(columns=['id_hash256', 'agg_text'])
    agg_text_df.to_csv(filename)
    
    for i, id in tqdm(enumerate(url_ids), total=len(url_ids)):
        agg_text = aggregate_interaction_text(id, **kwargs)
        df = pd.DataFrame({'id_hash256':id,'agg_text':agg_text}, index=[i])
        df.to_csv(filename, header=None, mode="a")


def get_metadata(filename=METADATA_FILE):
    return pd.read_json(filename)

    