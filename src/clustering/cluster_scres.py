import pandas as pd
import numpy as np

cluster_files = os.path.join(root, 'data/cluster_generated_reduced.json')
suspicion_scores = os.path.join(root, 'data/20221205-suspicion-scores.csv')

ROOT = 'drive/MyDrive/CDS_Capstone_2022_Fall'
metadata = pd.read_json(os.path.join(ROOT, 'data/capstone_url_metadata.json'))
metadata_hc = pd.read_json(os.path.join(ROOT, 'data/capstone_url_metadata_hc.json'))

og_meta=metadata[['url', 'id_hash256']]

merged_urls_meta=pd.concat([og_meta, metadata_hc], axis=0)
merged_urls_meta.head()

#input cluster_files, suspicion_scores
def get_cluster_scores(cluster_json, scores):
    """takes in json file and outputs a dataframe of nan clusters and scored clusters"""
    cluster_df=pd.read_json(cluster_json)
    ss_df=pd.read_csv(scores)
    #merge original urls with scores
    urls_ss=pd.merge(ss_df, merged_urls_meta, how='inner') 
    avg_cluster_susscores={}
    
    for i in range(len(updated_cluster.columns)):
      ith_cluster=updated_cluster.iloc[:,i].to_frame()
      ith_cluster=ith_cluster.rename(columns={i:'url'})
      #merge each cluster with its scores
      merged_urls=pd.merge(ith_cluster, urls_ss, how='inner')
      avg_sus_score=merged_urls.score.mean()
      avg_cluster_susscores[i]=avg_sus_score

    acs=pd.DataFrame(avg_cluster_susscores, index=[0])
    mod_acs=acs.T
    mod_acs=mod_acs.reset_index()
    mod_acs=mod_acs.rename(columns={'index':'cluster', 0:'avg_scores'})
    mod_acs=mod_acs.sort_values(by='avg_scores', ascending=False)
    rm_na=mod_acs.copy()
    nan_clusters=rm_na[rm_na.avg_scores.isna()]
    scored_clusters=rm_na[~rm_na.avg_scores.isna()]
    return nan_clusters, scored_clusters 
