import pandas as pd
import tweepy
import logging
from tweepy.auth import OAuthHandler
from src.twitter_api import config
from tqdm import tqdm

# import botometer

import json
from datetime import datetime
from time import sleep


# TODO: are there other features we can collect 
USER_FEATURES = [
    'verified', 
    'statuses_count',
    'created_at', 
    'followers_count',
    'friends_count',
    'protected'
] 

# all features (excluding no longer supported (deprecated) attributes)
USER_FEATURES_FULL = ['id', 'id_str', 'name', 'screen_name', 
                    'location', 'url', 'description', 'protected', 'verified', 
                    'followers_count', 'friends_count', 'listed_count', 'favourites_count', 'statuses_count', 
                    'created_at', 'profile_banner_url', 'profile_image_url_https', 
                    'default_profile', 'default_profile_image', 'withheld_in_countries']


def create_tweepy_auth(twitter_app_auth):
    
    auth = OAuthHandler(
        twitter_app_auth['consumer_key'], 
        twitter_app_auth['consumer_secret']
    )
    auth.set_access_token(
        twitter_app_auth['access_token'], 
        twitter_app_auth['access_token_secret']
    )
    return auth 


class ProfileScraper:
    """
    
    scrape profile of twitter users using Twitter API
    """

    def __init__(self, twitter_app_auth=None, wait_on_rate_limit=True, **kwargs):
        
        if twitter_app_auth is None:
            twitter_app_auth = config.twitter_app_auth
        self.auth = create_tweepy_auth(twitter_app_auth)
        self.api = tweepy.API(self.auth, wait_on_rate_limit=wait_on_rate_limit, **kwargs)
    
    
    def check_status(self, screen_name, features=USER_FEATURES):
        """
        
        Verify whether a user exists, or is suspended and 
        scrape basic information about the user
        """
        outputs = {key:None for key in features}
        
        try:
            user = self.api.get_user(screen_name=screen_name)
            outputs['status'] = 'normal'
            
            for feature in features:
                try: # check if user_feature exists 
                    outputs[feature] = getattr(user, feature)
                except Exception as e:
                    outputs[feature] = None
        
        # except tweepy.TweepyException as error: # tweepy version 4.0
        #     error_code = error.api_codes[0]
        except tweepy.error.TweepError as error: # tweepy version 3.0
            error_code = error.args[0][0]['code']    
            if error_code == 63:
                outputs['status'] = 'suspended'
            elif error_code == 50:
                outputs['status'] = 'not_found'
            outputs['screen_name'] = screen_name
                
        return outputs
    
    
    def check_suspension_status(self, screen_name):
        """
        
        return user status
            if user does not exist, status is None: 
        """
        return self.check_status(screen_name, features=[])
    
    
    def run(self, users, features=USER_FEATURES, save_file=None):
        """
        
        Scrape from profile for multiple input users 
        and save results as a CSV file 
        """
        for i, screen_name in enumerate(tqdm(users)):           
            outputs = self.check_status(screen_name, features)
            df = pd.DataFrame([outputs], index=[i])
            logging.debug(outputs)
            
            # add current row to existing csv file 
            if save_file is not None:
                df.to_csv(save_file, index=False, 
                          mode='w' if i == 0 else 'a', 
                          header=(i==0))
            # # add current row to existing dataframe
            # cumulative_df = df if i == 0 else cumulative_df.append(df)

        # return cumulative_df 

    def run_every15min(self, users, features=USER_FEATURES, save_file=None, num_requests=900):
        """
        
        Limit the number of requests for every 15 minutes 
        until all user profiles are scraped
        """
        print(f'Number of users: {len(users)}')
        print(f'Time needed: {len(users)//num_requests*15} min')

        i = 0
        while True:
            now = datetime.now()
            dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
            print(dt_string)

            # save as a new file every 15 minutes
            sub_save_file = save_file.split('.')
            sub_save_file[-2] += ('_' + str(i+1))

            sub_users_list = users[i*num_requests: (i+1)*num_requests]
            self.run(sub_users_list, features, '.'.join(sub_save_file))

            i += 1
            if i * num_requests >= len(users):
                return # finish

            sleep(60*15)  # Wait for 15 minutes
  
