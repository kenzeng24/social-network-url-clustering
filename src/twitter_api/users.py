import tweepy
import config
# import botometer
from tweepy.auth import OAuthHandler

# TODO: are there other features we can collect 
USER_FEATURES = [
    'verified', 
    'statuses_count',
    'created_at', 
    'followers_count',
    'friends_count',
    'protected'
] 


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

    def __init__(self, twitter_app_auth=None, **kwargs):
        
        if twitter_app_auth is None:
            twitter_app_auth = config.twitter_app_auth
        
        self.auth = create_tweepy_auth(twitter_app_auth)
        self.api = tweepy.API(self.auth, **kwargs)
    
    
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
        
        except tweepy.error.TweepError as error:
            
            error_code = error.args[0][0]['code']
            if error_code == 63:
                outputs['status'] = 'suspended'
                
        return outputs
    
    
    def check_suspension_status(self, screen_name):
        """
        return user status
            if user does not exist, status is None: 
        """
        return self.check_status(screen_name, features=[])
  
