import logging
import os
import requests
import websockets.exceptions

from cryptoxlib.version_conversions import async_run
from cryptoxlib.CryptoXLib import CryptoXLib
from cryptoxlib.clients.bitforex import enums
from cryptoxlib.Pair import Pair
from cryptoxlib.clients.bitforex.BitforexWebsocket import OrderBookSubscription, TradeSubscription, TickerSubscription, Ticker24hSubscription
from cryptoxlib.version_conversions import async_run

import pandas as pd
import sqlalchemy as db

LOG = logging.getLogger("cryptoxlib")
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.StreamHandler())

insert_new_element = (
    "INSERT INTO prices (price, amount, time, tid, direction, fk_currency_pair) VALUES (%s, %s, %s, %s, %s, %s)"
)

print(f"Available loggers: {[name for name in logging.root.manager.loggerDict]}\n")


async def trade_update(response: dict) -> None:
    print(f"Callback trade_update: []")
    parameters = response['param']['businessType'].split('-')[1:]
    pair = parameters[1].upper() + parameters[0].upper().replace(" ", "")
    fk_selection = currency_pairs[currency_pairs['symbol'] == pair]
    if not fk_selection.empty:
        fk_currency_pair = fk_selection.id.array[0]
        if 'data' in response:
            for item in response['data']:
                connection.execute(insert_new_element, (item['price'], item['amount'], item['time'], item['tid'], item['direction'], fk_currency_pair))
    else:
        print(f"Pair not found '{pair}'")


async def order_book_update(response: dict) -> None:
    print(f"Callback order_book_update: [{response}]")


async def order_book_update2(response: dict) -> None:
    print(f"Callback order_book_update2: [{response}]")


async def ticker_update(response: dict) -> None:
    print(f"Callback ticker_update: [{response}]")


async def ticker24_update(response: dict) -> None:
    print(f"Callback ticker24_update: [{response}]")


async def run(filtered_symbolsr):
    # to retrieve your API/SEC key go to your bitforex account, create the keys and store them in
    # BITFOREXAPIKEY/BITFOREXSECKEY environment variables
    api_key = os.environ.get('BITFOREXAPIKEY', "726bfa7cfc15ddec44aee8188bfbdb38")
    sec_key = os.environ.get('BITFOREXSECKEY', "bf6213977fefdfde03924b380eeed4ae")

    bitforex = CryptoXLib.create_bitforex_client(api_key, sec_key)
    size = "200"

    # Bundle several subscriptions into a single websocket
    subscription = [TradeSubscription(pair=Pair(item[0], item[1]), size="100", callbacks=[trade_update]) for item in filtered_symbols[:2]]
    bitforex.compose_subscriptions([
        TradeSubscription(pair=Pair("BTC", "USDT"), size=size, callbacks=[trade_update]),
        TradeSubscription(pair=Pair("DOGE", "USDT"), size=size, callbacks=[trade_update]),
    ])

    bitforex.compose_subscriptions([
        TradeSubscription(pair=Pair("MATIC", "BTC"), size=size, callbacks=[trade_update]),
        TradeSubscription(pair=Pair("XRP", "USDT"), size=size, callbacks=[trade_update]),
    ])

    bitforex.compose_subscriptions([
        TradeSubscription(pair=Pair("ETC", "USDT"), size=size, callbacks=[trade_update]),
        TradeSubscription(pair=Pair("BCH", "USDT"), size=size, callbacks=[trade_update]),
    ])

    bitforex.compose_subscriptions([
        TradeSubscription(pair=Pair("BCH", "BTC"), size=size, callbacks=[trade_update]),
        TradeSubscription(pair=Pair("ADA", "USDT"), size=size, callbacks=[trade_update]),
    ])

    bitforex.compose_subscriptions([
        TradeSubscription(pair=Pair("TMCN", "BTC"), size=size, callbacks=[trade_update]),
        TradeSubscription(pair=Pair("DOGE", "BTC"), size=size, callbacks=[trade_update]),
    ])

    for item in range(100):
        try:
            # Execute all websockets asynchronously
            await bitforex.start_websockets()
        except websockets.exceptions.ConnectionClosedOK as exc:
            print("Problem detected {exc}")

    await bitforex.close()


if __name__ == "__main__":
    # Connect with the MySQL Server
    engine = db.create_engine('mysql+pymysql://feeder:qwerty@localhost/crypto_assets')
    with engine.connect() as connection:
        metadata = db.MetaData()
        currency_pairs = pd.read_sql_table("currency_pairs", connection)

        actual_pair = currency_pairs[currency_pairs['symbol'] == "BTCUSDT"]

        response = requests.get("https://api.bitforex.com/api/v1/market/symbols")
        data = response.json()
        symbols = [item['symbol'] for item in data['data']]
        filtered_symbols = []
        allowed_symbols = ['btc', 'eth']
        for item in symbols:
            for allowed_symbol in allowed_symbols:
                if allowed_symbol in item:
                    splitted_symbols = item.split('-')
                    filtered_symbols.append((splitted_symbols[1].upper(), splitted_symbols[2].upper()))
        print(filtered_symbols)

        async_run(run(filtered_symbols))
