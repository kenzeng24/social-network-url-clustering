import pandas as pd 
import numpy as np 
import json, re, os 


ROOT = 'drive/MyDrive/CDS_Capstone_2022_Fall'
METADATA_FILE = os.path.join(ROOT, 'data/capstone_url_metadata.json')
AGGREGATE_DATA_PATH = os.path.join(ROOT, 'data/data')
EMBEDDING_PATH = os.path.join(ROOT, 'embedding')


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
    output_text = ""
    for platform_data in result['result'].values():
        for data in platform_data:
            if data['i'] > i_min:
                output_text += clean_text(data['d'].lower()) + " "
    return output_text[:-1] # remove empty space at the end


def aggregate_text(metadata, **kwargs):
    return metadata.id_hash256.apply(
        lambda id: aggregate_interaction_text(id, **kwargs))


def get_metadata(filename=METADATA_FILE):
    return pd.read_json(filename)

