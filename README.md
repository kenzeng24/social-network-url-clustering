# User-Discussion Driven Approach to Detecting Cross-Platform Coordinated Influence Campaigns

Fake news has become a widespread problem on social media platforms, with potentially serious consequences for individuals and society as a whole. The spread of fake news can lead to the dissemination of false information, damage reputations, and even undermine the integrity of elections. A user-context-driven approach to detect fake news and determine its veracity involves taking into account the specific context in which the news is being shared. This involves considering factors such as the way people share it online. By taking these factors into account, a user-context-driven approach provides a more nuanced and accurate assessment of the news than a simple fact-checking approach that only considers the veracity of individual claims. In this paper, we will explore machine-learning approaches to detecting fake news on multiple social platforms. We achieved an AUC of 0.92 without relying on the website's contents. Our approach shows that the surrounding user interactions can be a strong substitute for evaluating the source's credibility. 

## Data Collection: Information Tracer

The data is collected from [Information tracer](https://informationtracer.com). Information Tracer is a cross-platform data collection API created and implemented by Zhouhan Chen. It takes a URL as input and collects posts across multiple social media platforms sharing that one URL. Each collected post contains information about its text content, the publisher's username, the timestamp, and the number of interactions (likes, favorites, etc.).

Using Information Tracer, we collected posts from 5 major social media platforms (Facebook, Reddit, Youtube, Telegram, and Twitter) for all 71k+ URLs we have. It provides various forms of user interaction data with the news across multiple platforms, which well aligns with our objective. For access to the original unprocessed data, please contact the Information Tracer team at zhouhan@safelink.network. 

<p align="center">
  <img src="https://github.com/kenzeng24/social-network-url-clustering/blob/main/figures/data.png">
</p>


## Fake News Labels 

[Media Bias/Fact Checker (MBFC)](https://mediabiasfactcheck.com) is an open-source and free online website that produces biased/factual ratings on media websites. Some main features it provides include factual reporting accuracy, the level of credibility, traffic/popularity, and bias in terms of political standing for each media site.
 
For the purpose of building a fake news detection model, we primarily used the Credibility ratings from MBFC, which measures the credibility of a site. As an outcome of using these categories to train our model, we intend to generate labels for URLs or domains that do not have a score. To create our own binary classification scoring function, we centered our attention on the "High" and "Low" Credibility labels and mapped them to new values. 1 pertains to sites with a lack of credibility and is viewed as suspicious and the remaining sites that have high credibility and are not suspicious are given a label of 0. We can define suspicious sites as having a history of falsifying information. Given the intent of our model, the two categories provide information in regard to suspicious domains. In addition, we used MBFC as the primary source that sets the path toward collecting additional features.

## getting started

```bash
git clone https://github.com/kenzeng24/social-network-url-clustering.git
```

```bash
cd /path/to/this/directory
export PYTHONPATH=$(pwd):$PYTHONPATH
```

To load the functions from github repo into jupyter notebook 
```python
import sys
sys.path.append('<path to social-network-url-clustering>')
```

## Usage 

### Data Scraping

[Twitter API](https://developer.twitter.com/en/docs/twitter-api) was used to collect user statistics of the Twitter users retrieved by Information Tracer. In total, 70937 unique user statistics were collected. These users have either posted at least one original tweet or retweeted the news URLs in our dataset for more than 20 times. Each user has 17 features, and we utilized 8 of them in our model training. 

Compared to other social media platforms,  Twitter has a readily accessible API. For example, the feature of checking whether or not a user is verified is either never available or has been deprecated among all the other four platforms. In addition, Twitter has the highest volume of posts in relation to our URLs. In our machine learning pipeline, we will use these collected features to describe the population of users that reposted a particular URL. To use tweepy to scrape inforamtion about the twitter users:

```python
from src.twitter_api.users import ProfileScraper
from src.twitter_api import config

# update config with your keys 
config.twitter_app_auth = {
    'consumer_key': 'XXXX',
    'consumer_secret': 'XXXX',
    'access_token': 'XXXX',
    'access_token_secret': 'XXXX',
}

scraper = ProfileScraper()
users = ["AnjneyMidha", "oddlikepie", "RealDonaldTrump"]
scraper.run(users, save_file=None)
```

### Data Processing 

To aggregate interaction text for a list of urls from metadatajson: 

```python 
from src.data_loading import interactions 
from src.data_loading.interactions import aggregate_text, get_metadata

# get metadatafile and save outputs to <agg_output_filename>
metadata = get_metadata('<path-to-metedata>')
aggregate_text(metadatafile.id_hash256, '<path-to-output>')
```

To convert aggregate text into TFIDF vectors and also 

```python 
import scipy.sparse import save_npz
from src.data_loading.aggregated_text import load_aggregated_data

)
metadata_aggregated_balanced = load_aggregated_data() 

tfidf_vectors, vectorizer = train_tfidf_vectorizer(
    metadata_aggregated_balanced.agg_text,
    filename='<tfidf-vectorizer-filename>'
)
scipy.sparse.save_npz('<tfidf-matrix-filename>', tfidf_vectors)
```

### Model training 

To load the traininng data: 

```python 
X, y, urls = get_balanced_tfidf_data()
```

And to get URLs unlabeled by MBFC: 

```python 
X_unlabeled, urls_unlabeled = get_unabeled_urls()
```

To perform cross validation:

```python 
params = logistic_regression_cv(random_state=824)
```

## 


