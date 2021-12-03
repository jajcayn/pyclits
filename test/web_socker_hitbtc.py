import logging
import os
import uuid
import requests

import sqlalchemy as db

from cryptoxlib.CryptoXLib import CryptoXLib
from cryptoxlib.Pair import Pair
from cryptoxlib.clients.hitbtc.HitbtcWebsocket import TickerSubscription, OrderbookSubscription, TradesSubscription, \
    AccountSubscription, ClientWebsocketHandle, CreateOrderMessage, CancelOrderMessage
from cryptoxlib.clients.hitbtc import enums
from cryptoxlib.version_conversions import async_run

LOG = logging.getLogger("cryptoxlib")
LOG.setLevel(logging.DEBUG)
LOG.addHandler(logging.StreamHandler())

print(f"Available loggers: {[name for name in logging.root.manager.loggerDict]}\n")


async def order_book_update(response: dict) -> None:
    print(f"Callback order_book_update: [{response}]")


async def ticker_update(response: dict) -> None:
    print(f"Callback ticker_update: [{response}]")


async def trade_update(response: dict) -> None:
    print(f"Callback trade_update: [{response}]")


async def account_update(response: dict, websocket: ClientWebsocketHandle) -> None:
    print(f"Callback account_update: [{response}]")

    # as soon as account channel subscription is confirmed, fire testing orders
    if 'id' in response and 'result' in response and response['result'] == True:
        await websocket.send(CreateOrderMessage(
            pair = Pair('ETH', 'BTC'),
            type = enums.OrderType.LIMIT,
            side = enums.OrderSide.BUY,
            amount = "1000000000",
            price = "0.000001",
            client_id = str(uuid.uuid4())[:32]
        ))

        await websocket.send(CancelOrderMessage(
            client_id = "client_id"
        ))


async def run():
    api_key = os.environ.get('HITBTCAPIKEY', 'tSvtJay35HBVKaz2EGkKNSHR_55FD4PE')
    sec_key = os.environ.get('HITBTCSECKEY', 'KCnNs3pyoENvDd4zLor6OsmFk9Fg7NEg')

    client = CryptoXLib.create_hitbtc_client(api_key, sec_key)

    # Bundle several subscriptions into a single websocket
    client.compose_subscriptions([
        #TickerSubscription(pair=Pair("BTC", "USD"), callbacks = [ticker_update]),
        TradesSubscription(pair=Pair("BTC", "USD"), limit=100, callbacks=[trade_update]),
        TradesSubscription(pair=Pair("ETH", "BTC"), limit=100, callbacks=[trade_update])
    ])

    # Execute all websockets asynchronously
    await client.start_websockets()

if __name__ == "__main__":
    response = requests.get("https://api.hitbtc.com/api/3/public/symbol")
    if response.ok:
        symbols = response.json()
        print([key for key, item in symbols.items()])

        engine = db.create_engine('mysql+pymysql://feeder:qwerty@localhost/crypto_assets')
        with engine.connect() as connection:
            metadata = db.MetaData(bind=connection)
            metadata.reflect()
            currency = metadata.tables['currency']

            for key, item in symbols.items():
                if item['type'] == "spot":
                    for symbol in (item['base_currency'], item['quote_currency']):
                        try:
                            db_stsm = (db.select(currency.columns.id).where(currency.columns.symbol == f'{symbol}') )
                            result = connection.execute(db_stsm).fetchall()
                            if not result:
                                db_stsm = (db.insert(currency).values(symbol=symbol))
                                connection.execute(db_stsm)
                        except Exception as exc:
                            print(f"{exc} {symbol}")
                else:
                    print(f"Special case {item}")

            db_stsm = (db.select(currency.columns.id, currency.columns.symbol))
            result = connection.execute(db_stsm).fetchall()
            currency_table = {item[1]: item[0] for item in result}

            currency_pairs = metadata.tables['currency_pairs']
            for key, item in symbols.items():
                if item['type'] == 'spot':
                    db_stsm = (db.select(currency_pairs.columns.id).where(currency_pairs.columns.symbol == f'{key}'))
                    result = connection.execute(db_stsm).fetchall()
                    if not result:
                        db_stsm = (db.insert(currency_pairs).values(
                            symbol=key,
                            base_currency=currency_table[item['base_currency']],
                            quote_currency=currency_table[item['quote_currency']])
                        )
                        connection.execute(db_stsm)
                    else:
                        print(f"Already exist {key}")
    else:
        print(f"Error {response}")

    async_run(run())
