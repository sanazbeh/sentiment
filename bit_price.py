import requests

url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
response = requests.get(url)
data = response.json()

btc_price = data["price"]
print(f"Bitcoin Price on Binance: ${btc_price}")