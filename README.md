# social-network-url-clustering

```bash
cd /path/to/this/directory
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Usage 

### Data Scraping

Use tweepy to scrape inforamtion about the twitter users:

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


