import textstat 
import json

root='drive/MyDrive/CDS_Capstone_2022_Fall'

filename=''

def text_score_generator(filename):
	macld = os.path.join(root, 'data/'+f'filename')
	macld_df=pd.read_json(macld)
	
	#Flesh Reading Ease Score
	macld_df['FLRE_scre']=macld_df.apply(lambda x: textstat.flesch_reading_ease(x))

	#Grade level agg. test of text
	macld_df['grade_scre']=macld_df.apply(lambda x: textstat.text_standard(x))	

	return macld_df 
		
