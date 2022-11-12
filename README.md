# social-network-url-clustering

```bash
cd /path/to/this/directory
export PYTHONPATH=$(pwd):$PYTHONPATH
```

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


