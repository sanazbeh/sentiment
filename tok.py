import requests

# Your Bearer Token from X API
BEARER_TOKEN = "1574167072522780673-3DPYnEV7Uj5MBiaUWe0r5syJFBip5O"

def get_tweets():
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    url = "https://api.twitter.com/2/tweets/search/recent?query=bitcoin&max_results=10"
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        tweets = response.json()
        return tweets
    else:
        print("Error:", response.status_code, response.text)
        return None

tweets = get_tweets()
print(tweets)