# real_time_social_media_analysis.py

import tweepy
import time
import openai
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

def authenticate_twitter_api(consumer_key, consumer_secret, access_token, access_token_secret):
    """
    Authenticate with the Twitter API using the provided credentials.

    Args:
        consumer_key (str): Twitter API consumer key.
        consumer_secret (str): Twitter API consumer secret.
        access_token (str): Twitter API access token.
        access_token_secret (str): Twitter API access token secret.

    Returns:
        tweepy.API: Authenticated Tweepy API object.
    """
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)
    return api

def fetch_real_time_tweets(api, query, lang='en', tweet_mode='extended', count=100):
    """
    Fetch real-time tweets based on the specified query.

    Args:
        api (tweepy.API): Authenticated Tweepy API object.
        query (str): Search query to fetch tweets.
        lang (str): Language code for the tweets (default: 'en' for English).
        tweet_mode (str): Tweet mode to fetch full-text tweets (default: 'extended').
        count (int): Number of tweets to fetch per request (default: 100).

    Yields:
        str: Full text of each tweet.
    """
    while True:
        try:
            for tweet in tweepy.Cursor(api.search_tweets, q=query, lang=lang, tweet_mode=tweet_mode, count=count).items():
                yield tweet.full_text
            time.sleep(60)  # Wait for 60 seconds before making the next request
        except tweepy.TweepError as e:
            print(f"Error: {str(e)}")
            time.sleep(60)  # Wait for 60 seconds before retrying

def preprocess_tweets(tweets):
    """
    Preprocess the tweets by removing mentions, URLs, and hashtags.

    Args:
        tweets (list): List of tweet texts.

    Returns:
        list: List of preprocessed tweet texts.
    """
    preprocessed_tweets = []
    for tweet in tweets:
        # Remove mentions
        tweet = ' '.join(word for word in tweet.split() if not word.startswith('@'))
        # Remove URLs
        tweet = ' '.join(word for word in tweet.split() if not word.startswith('http'))
        # Remove hashtags
        tweet = ' '.join(word for word in tweet.split() if not word.startswith('#'))
        preprocessed_tweets.append(tweet)
    return preprocessed_tweets

def analyze_sentiment(tweets, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
    """
    Perform sentiment analysis on the tweets using a pre-trained BERT model.

    Args:
        tweets (list): List of tweet texts.
        model_name (str): Name of the pre-trained BERT model for sentiment analysis (default: 'distilbert-base-uncased-finetuned-sst-2-english').

    Returns:
        list: List of sentiment labels for each tweet.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
        sentiments = classifier(tweets)
        return [sentiment['label'] for sentiment in sentiments]
    except Exception as e:
        print(f"Error during sentiment analysis: {str(e)}")
        return []

def generate_insights(tweets, engine='davinci', max_tokens=100, n=1, stop=None, temperature=0.7):
    """
    Generate insights from the tweets using the OpenAI GPT-3 model.

    Args:
        tweets (list): List of tweet texts.
        engine (str): OpenAI GPT-3 engine to use for text generation (default: 'davinci').
        max_tokens (int): Maximum number of tokens to generate in the response (default: 100).
        n (int): Number of responses to generate (default: 1).
        stop (list): List of tokens to stop the generation at (default: None).
        temperature (float): Temperature for sampling the next token (default: 0.7).

    Returns:
        list: List of generated insights.
    """
    openai.api_key = "YOUR_API_KEY"  # Replace with your OpenAI API key
    prompt = f"Generate insights from the following tweets:\n\n{', '.join(tweets)}\n\nInsights:"
    try:
        response = openai.Completion.create(
            engine=engine,
            prompt=prompt,
            max_tokens=max_tokens,
            n=n,
            stop=stop,
            temperature=temperature
        )
        return [choice.text.strip() for choice in response.choices]
    except openai.error.OpenAIError as e:
        print(f"OpenAI API Error: {str(e)}")
        return []

def main():
    consumer_key = "YOUR_CONSUMER_KEY"
    consumer_secret = "YOUR_CONSUMER_SECRET"
    access_token = "YOUR_ACCESS_TOKEN"
    access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"

    api = authenticate_twitter_api(consumer_key, consumer_secret, access_token, access_token_secret)

    query = "data science"
    batch_size = 10

    tweets = []
    for tweet in fetch_real_time_tweets(api, query):
        tweets.append(tweet)
        if len(tweets) == batch_size:
            preprocessed_tweets = preprocess_tweets(tweets)
            sentiments = analyze_sentiment(preprocessed_tweets)
            insights = generate_insights(preprocessed_tweets)

            print("Tweet Sentiments:")
            for tweet, sentiment in zip(tweets, sentiments):
                print(f"Tweet: {tweet}")
                print(f"Sentiment: {sentiment}")
                print()

            print("Generated Insights:")
            for insight in insights:
                print(insight)
                print()

            tweets = []  # Reset the tweets list for the next batch

if __name__ == "__main__":
    main()
