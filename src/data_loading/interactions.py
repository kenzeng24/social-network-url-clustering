import pandas as pd 
import numpy as np 
import json, re, os
import spacy, string, nltk

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
    output_text = ""
    for platform_data in result['result'].values():
        for data in platform_data:
            if data['i'] > i_min:
                output_text += clean_text(data['d'].lower()) + " "
            else:
                output_text = '<empty> '
    return output_text[:-1] # remove empty space at the end


def aggregate_text(metadata, **kwargs):
    return metadata.id_hash256.apply(
        lambda id: aggregate_interaction_text(id, **kwargs))


def get_metadata(filename=METADATA_FILE):
    return pd.read_json(filename)

#this will be used in CountVectorizer
def tokenize(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # translator that replaces punctuation with empty spaces
    return [stemmer.stem(i) for i in text.translate(translator).split()]  # stemmer and tokenizing into words

# Tokenize stop words to match 
stop_words = [tokenize(s)[0] for s in stop_words]
stop = stop_words 
full_stopwords = [tokenize(s)[0] for s in stop]

def n_grams_vectorizer(ngram=1, min_df=0.05, max_df=0.95, **kwargs):
    
    # TODO: add additional customization for vectorizer 
    vectorizer = CountVectorizer(
        stop_words=set(stop),
        analyzer="word",        # unit of features are single words rather than characters
        tokenizer=tokenize,      # function to create tokens
        ngram_range=(0,int(ngram)),       # change num of words co-located
        strip_accents='unicode', # remove accent characters
        min_df = min_df,           
        max_df = max_df, 
        **kwargs
    )           # only include words with maximum frequency of 0.95
    return vectorizer

def tfidf_vectorize(text_list,vectorizer=None, ngram=1, **kwargs):

    # convert text list to bag of words 
    if vectorizer is None:
        vectorizer = n_grams_vectorizer(ngram)
    bag_of_words = vectorizer.fit_transform(text_list)  # transform our corpus as a bag of words
    features = vectorizer.get_feature_names()  
    
    # convert bag of words to tfidf
    transformer = TfidfTransformer(
        norm = None, 
        smooth_idf = True, 
        sublinear_tf = True, **kwargs
    )
    tfidf = transformer.fit_transform(bag_of_words)
    return tfidf, features

def topic_generator(tfidf, features, num_topics=10, verbose=False, **kwargs):

    # Fitting LDA model
    lda = LatentDirichletAllocation(
        n_components = num_topics, 
        learning_method='online',
        random_state=42, **kwargs) #adjust n_components

    doctopic = lda.fit_transform(tfidf)

    # Displaying the top keywords in each topic
    ls_keywords = []
    for i,topic in enumerate(lda.components_):
        word_idx = np.argsort(topic)[::-1][:10]
        keywords = ', '.join(features[i] for i in word_idx)
        ls_keywords.append(keywords)
        if verbose:
            print(i, keywords) 
    return doctopic, ls_keywords

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
