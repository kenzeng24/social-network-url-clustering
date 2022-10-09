import spacy, string, nltk,os,pickle
import numpy as np 
import pandas as pd 

from src.data_loading import interactions

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn import preprocessing
from nltk import SnowballStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')

stemmer = SnowballStemmer("english")

# TODO: replace these with the actual files
TFIDF_FILE = "tfidf.pickle" 
METADATA_TFIDF = 'metadata_tfidf.json'
TOPIC_MODEL='topic_model.pickle'

def tokenize(text):
    # translator that replaces punctuation with empty spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  
    return [stemmer.stem(i) for i in text.translate(translator).split()]  

stop_words = set([tokenize(word)[0] for word in stopwords.words('english')])


def train_tfidf_vectorizer(
        text_list, 
        filename=TFIDF_FILE ,
        max_features=1000, stop_words=stop_words, **kwargs):
    
    assert not filename or not os.path.exists(filename), 'tfidf file already exists'
    
    vectorizer = TfidfVectorizer(
        analyzer='word', 
        strip_accents='unicode',
        tokenizer=tokenize,
        stop_words=stop_words,
        max_features=max_features, **kwargs)
    
    tfidf_vectors = vectorizer.fit_transform(text_list)
    if filename: pickle.dump(vectorizer, open(filename, "wb"))
    return tfidf_vectors, vectorizer


def tfidf_transform(text_list, filename=TFIDF_FILE):
    """load tfidf file from path """
    with open(filename, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer.transform(text_list)


def training_pipeline(filename=METADATA_TFIDF, 
                      tfidf_filename=TFIDF_FILE,
                      metadata_filename=interactions.METADATA_FILE, testing=False,**kwargs):
    """convert metadata into TFIDF vectors and store tfidf vectorizer"""
    metadata = interactions.get_metadata(filename=metadata_filename)
    if testing:
        metadata = metadata[:100]
        filename = 'testing-' + filename
        tfidf_filename = 'testing-' + tfidf_filename
    
    metadata['agg_text'] = aggregate_text(metadata, i_min=100)
    tfidf_vectors, vectorizer = train_tfidf_vectorizer(
        metadata['agg_text'], 
        filename=tfidf_filename, **kwargs)
    
    metadata['tfidf'] = tfidf_vectors.toarray().tolist()
    if testing: filename = 'testing-' + filename
    if filename and not os.path.exists(filename):
        metadata.to_json(filename)
    return metadata

                        
def topic_generator(tfidf, num_topics=10, verbose=False, 
                    filename='topic_model.pickle', **kwargs):
    """train topic model with TFIDF vectors"""
    assert not filename or not os.path.exists(filename), 'topic model already exists'
    # Fitting LDA model
    lda = LatentDirichletAllocation(
        n_components = num_topics, 
        learning_method='online',
        random_state=42, **kwargs) #adjust n_components
    
    doctopic = lda.fit_transform(tfidf)
    if filename: pickle.dump(lda, open(filename, "wb"))
    return doctopic, lda


def topic_transform(tfidf_vectors, filename=TOPIC_MODEL):
    """load topic model and transform tfidf vectors in topic vectors """
    with open(filename, 'rb') as f:
        lda = pickle.load(f)
    return lda.transform(tfidf_vectors)


def get_topics(topic_model_filename=TOPIC_MODEL, tfidf_filename=TFIDF_FILE):
    """load topic model, TFIDF from pickle and get topics"""
    with open(topic_model_filename, 'rb') as f: lda = pickle.load(f)
    with open(tfidf_filename, 'rb') as f: vectorizer = pickle.load(f)
    features = {v:k for k,v in vectorizer.vocabulary_.items()}   

    # Displaying the top keywords in each topic
    ls_keywords = []
    for i,topic in enumerate(lda.components_):
        word_idx = np.argsort(topic)[::-1][:10]
        keywords = ', '.join(features[i] for i in word_idx)
        ls_keywords.append(keywords)
        print(i, keywords) 
    return ls_keywords  


def analyze_all_clusters(cluster_file=interactions.CLUSTER_FILE):
    pass 


def generate_cluster_vector(cluster, filename=interactions.METADATA_FILE):
    '''
    Inputs:
    cluster: list of URLs
    filename: original metadata file 
    ngram: number of ngrams to use 
    num_topics: number of topics to generate 
    i_min: minimum number of interactions
    n_len: threshold length of text to consider in the metadata set
    '''
    
    # extract tfidf vectors from 'metadata_tfidf.csv'
    pass 
    