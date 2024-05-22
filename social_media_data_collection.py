import tweepy
import json

API_KEY = "your_api_key"
API_SECRET_KEY = "your_api_secret_key"
ACCESS_TOKEN = "your_access_token"
ACCESS_TOKEN_SECRET = "your_access_token_secret"

auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def fetch_tweets(query, count=100):
    tweets = api.search_tweets(q=query, count=count, lang='en', tweet_mode='extended')
    tweets_data = []
    for tweet in tweets:
        tweets_data.append({
            'id': tweet.id_str,
            'created_at': tweet.created_at.isoformat(),
            'full_text': tweet.full_text,
            'user': tweet.user.screen_name
        })
    return tweets_data

def save_tweets(tweets, file_name):
    with open(file_name, 'w') as file:
        json.dump(tweets, file, indent=4)

# Example usage
if __name__ == "__main__":
    try:
        tweets = fetch_tweets("#wildfire", 100)
        save_tweets(tweets, "tweets.json")
    except tweepy.TweepyException as e:
        print(f"Failed to fetch tweets: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
