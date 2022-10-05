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

def cluster_tm_analysis(cluster_json=CLUSTER_FILE, filename=METADATA_FILE, ngram=1, num_topics=10, i_min=100, n_len=0):
  '''
  Inputs:
  cluster_json: CLUSTER_FILE condensed cluster file in json format
  filename: original metadata file 
  ngram: number of ngrams to use 
  num_topics: number of topics to generate 
  i_min: minimum number of interactions
  n_len: threshold length of text to consider in the metadata set
  '''
  with open(CLUSTER_FILE, 'rb') as f:
    cluster_json = json.load(f)

  df_metadata=get_metadata(filename=METADATA_FILE)

  cluster_topics=[]

  for cluster in cluster_json:
    subset=df_metadata[df_metadata['url'].isin(cluster)] #filter original metadata with all the urls from a specific cluster

    subset['agg_text']=aggregate_text(subset, i_min=i_min)  #takes subset['id_hash256'] and performs the aggregations

    filtered_data = subset[subset.agg_text.apply(len)>n_len]

    tfidf, features = tfidf_vectorize(filtered_data.agg_text, ngram=ngram) #change ngrams here? predefined vocab can be adjusted

    outputs=topic_generator(tfidf, features, num_topics=num_topics)

    stored_outputs=outputs[1]
  
    cluster_topics.append(stored_outputs)

  return cluster_topics
