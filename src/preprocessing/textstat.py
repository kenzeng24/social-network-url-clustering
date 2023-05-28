import textstat 
import os
import pandas as pd 
from src.data_loading.interactions import ROOT


def text_score_generator(filename):
	macld = os.path.join(ROOT, 'data', filename)
	macld_df=pd.read_json(macld)
	
	#Flesh Reading Ease Score
	macld_df['FLRE_scre']=macld_df['agg_text'].apply(lambda x: textstat.flesch_reading_ease(x))
	
	#coleman_liau_index; higher index corresponds to higher complexity
	macld_df['Cole_Idx']=macld_df['agg_text'].apply(lambda x: textstat.coleman_liau_index(x))

	#Grade level agg. test of text
	macld_df['grade_scre']=macld_df['agg_text'].apply(lambda x: textstat.text_standard(x))	

	return macld_df 
		
