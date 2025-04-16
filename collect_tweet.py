import tweepy

# جایگزین کن با API Keys خودت
API_KEY = "IklKfnYofRol2FTiVQjdebDSa"
API_SECRET = "9kyc0sHSSjQsfMvcBqWJWV9sWtIcsrtSFc4GcxbcYaAdzodlQA"
ACCESS_TOKEN = "1574167072522780673-b895XgNabiXUdp8gwj2lztUIbe13BY"
ACCESS_TOKEN_SECRET = "eKRo7mdQ0ip4Vejur0sCiDPF4JQki0Z9z1KqT062GvbGU"

# احراز هویت
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

def get_bitcoin_tweets():
    tweets = api.search_tweets(q="Bitcoin", lang="en", count=10)
    return [tweet.text for tweet in tweets]

btc_tweets = get_bitcoin_tweets()
print(btc_tweets)