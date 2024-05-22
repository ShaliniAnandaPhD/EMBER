import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')

def load_tweets(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
    return text.lower()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def preprocess_tweets(tweets):
    preprocessed_tweets = []
    for tweet in tweets:
        clean_tweet = clean_text(tweet['full_text'])
        preprocessed_tweet = preprocess_text(clean_tweet)
        preprocessed_tweets.append({
            'id': tweet['id'],
            'created_at': tweet['created_at'],
            'preprocessed_text': preprocessed_tweet,
            'user': tweet['user']
        })
    return preprocessed_tweets

def save_preprocessed_tweets(tweets, file_name):
    with open(file_name, 'w') as file:
        json.dump(tweets, file, indent=4)

# Example usage
if __name__ == "__main__":
    try:
        raw_tweets = load_tweets("tweets.json")
        preprocessed_tweets = preprocess_tweets(raw_tweets)
        save_preprocessed_tweets(preprocessed_tweets, "preprocessed_tweets.json")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
