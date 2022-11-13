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

from src.twitter_api.users import ProfileScraper
from src.twitter_api import config

# update config with your keys 
config.twitter_app_auth = {
    'consumer_key': 'XXXX,
    'consumer_secret': 'XXXX',
    'access_token': 'XXXX',
    'access_token_secret': 'XXXX',
}
# 
scraper = ProfileScraper()
users = ["AnjneyMidha", "oddlikepie", "RealDonaldTrump"]
df = scraper.run(users, save_file=None)
```

### Data Processing 

To aggregate interaction text for a list of urls from metadatajson: 

```python 
from src.data_loading import interactions 
from src.data_loading.interactions import aggregate_text, get_metadata

# example metadata and output path 
metadata_filename = os.path.join(interactions.ROOT, 'data/capstone_url_metadata.json')
agg_output_filename = os.path.join(interactions.ROOT, 'data/metadata-aggregated.csv')

# get metadatafile and save outputs to <agg_output_filename>
metadata = get_metadata(filename)
aggregate_text(metadatafile.id_hash256, agg_output_filename)
```

To convert aggregate text into TFIDF vectors and also 

```python 
import scipy.sparse import save_npz

from src.data_loading.aggregated_text import load_aggregated_data

tfidf_vectorizer_filename = os.path.join(
    interactions.ROOT, 
    "models/balanced-dataset-tfidf-vectorizer.pickle"
)
tfidf_matrix_filename =  os.path.join(
    interactions.ROOT, 
    "data/balanced-dataset-tfidf."
)
metadata_aggregated_balanced = load_aggregated_data() 

tfidf_vectors, vectorizer = train_tfidf_vectorizer(
    metadata_aggregated_balanced.agg_text,
    filename=tfidf_filename
)
scipy.sparse.save_npz(tfidf_matrix_filename, tfidf_vectors)
```

### Model training 

```python 
X,y,urls = get_tfidf_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(**result.best_params_).fit(X=X_train,y=y_train)
```

And to get URLs unlabeled by MBFC: 

```python 
X_unlabeled, urls_unlabeled = get_unabeled_urls()
```


