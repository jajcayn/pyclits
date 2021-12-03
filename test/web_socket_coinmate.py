import logging
import os

import requests
from cryptoxlib.CryptoXLib import CryptoXLib
from cryptoxlib.Pair import Pair
from cryptoxlib.clients.coinmate.CoinmateWebsocket import OrderbookSubscription, TradesSubscription, \
    UserOrdersSubscription, BalancesSubscription, UserTradesSubscription, UserTransfersSubscription
from cryptoxlib.version_conversions import async_run

LOG = logging.getLogger("cryptoxlib")
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.StreamHandler())

print(f"Available loggers: {[name for name in logging.root.manager.loggerDict]}\n")


async def order_book_update(response: dict) -> None:
    print(f"Callback order_book_update: [{response}]")


async def trades_update(response: dict) -> None:
    print(f"Callback trades_update: [{response}]")


async def account_update(response: dict) -> None:
    print(f"Callback account_update: [{response}]")


async def run(currency_pairs):
    api_key = os.environ.get('COINMATEAPIKEY', 'D8Ftt6t1g3CLHk9_0mkjwCMdfXUbMQwxmVhAKkNujUM')
    sec_key = os.environ.get('COINMATESECKEY', '2WcboY618YHSLj7LTq3OOxFx_7XoT370hLl4VrNgWlc')
    user_id = os.environ.get('COINMATEUSERID', '89891')

    client = CryptoXLib.create_coinmate_client(user_id, api_key, sec_key)

    # Bundle several subscriptions into a single websocket

    subscriptions = [(TradesSubscription(Pair(pair['fromSymbol'], pair['toSymbol']), callbacks=[trades_update])) for pair in currency_pairs]
    client.compose_subscriptions(subscriptions)

    #client.compose_subscriptions([
    #    OrderbookSubscription(Pair("BTC", "EUR"), callbacks=[order_book_update]),
    #    TradesSubscription(pair=Pair("BTC", "EUR"), callbacks=[trades_update]),
    #])

    # Execute all websockets asynchronously
    await client.start_websockets()

if __name__ == "__main__":
    response = requests.get("https://coinmate.io/api/products")
    dataset = response.json()
    currency_pairs = dataset['data']
    pairs = [currency_pair['id'] for currency_pair in currency_pairs]
    print(pairs)

    async_run(run(currency_pairs))
