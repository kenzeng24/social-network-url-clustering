import spacy, string, nltk
import numpy as np 
import pandas as pd 

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

#this will be used in CountVectorizer
def tokenize(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))  # translator that replaces punctuation with empty spaces
    return [stemmer.stem(i) for i in text.translate(translator).split()]  # stemmer and tokenizing into words

# Tokenize stop words to match 
stop_words = [tokenize(s)[0] for s in stop_words]
stop = stop_words 
full_stopwords = [tokenize(s)[0] for s in stop]


def n_grams_vectorizer(
    ngram, 
    stop_words=set(stop),
    analyzer="word",        # unit of features are single words rather than characters
    tokenizer=tokenize,      # function to create tokens
    ngram_range=(0,int(ngram)),       # change num of words co-located
    strip_accents='unicode', # remove accent characters
    min_df = 0.05,           # only include words with minimum frequency of 0.05
    max_df = 0.95,  **kwargs):
    
    # TODO: add additional customization for vectorizer 
    vectorizer = CountVectorizer(
        stop_words=set(stop),
        analyzer="word",        # unit of features are single words rather than characters
        tokenizer=tokenize,      # function to create tokens
        ngram_range=(0,int(ngram)),       # change num of words co-located
        strip_accents='unicode', # remove accent characters
        min_df = 0.05,           # only include words with minimum frequency of 0.05
        max_df = 0.95, 
        **kwargs
    )           # only include words with maximum frequency of 0.95
    return vectorizer


# separate TFIDF from LDA
def tfidf_vectorize(text_list,vectorizer=None, ngram=1, **kwargs):

    if vectorizer is None:
        vectorizer = n_grams_vectorizer(ngram)
    bag_of_words = vectorizer.fit_transform(text_list)  # transform our corpus as a bag of words
    features = vectorizer.get_feature_names()  
    
    transformer = TfidfTransformer(
        norm = None, 
        smooth_idf = True, 
        sublinear_tf = True, **kwargs)
    
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