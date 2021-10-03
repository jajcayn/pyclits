import datetime
import pandas as pd
import sqlalchemy as db

if __name__ == "__main__":
    engine = db.create_engine('mysql+pymysql://feeder:qwerty@localhost/crypto_assets')
    with engine.connect() as connection:
        metadata = db.MetaData(bind=connection)
        currency_pairs = pd.read_sql_table("currency_pairs", connection)

        TABLE_EXISTS = connection.dialect.has_table(connection, "prices")
        metadata.reflect()
        prices = metadata.tables['prices']
        processed_price = metadata.tables['processed_price']

        for id_pair, symbol in currency_pairs.values:
            stmt = db.select([
                prices.columns.time,
                prices.columns.price,
                prices.columns.amount,
                prices.columns.tid
            ]
            ).where(
                prices.columns.fk_currency_pair == f'{id_pair}',
            ).order_by(prices.columns.time)

            results = connection.execute(stmt).fetchall()
            last_datetime_transaction = None
            total_amount = 0
            last_tid = 0
            record = 0
            for result in results:
                raw_time = result[0]
                datetime_transaction = datetime.datetime.fromtimestamp(raw_time/1000)
                price = float(result[1])
                amount = float(result[2])
                tid = result[3]
                if (last_datetime_transaction == datetime_transaction):
                    if (last_tid != tid):
                        total_amount += amount
                    else:
                        print(f"Duplication detected {result}")
                else:
                    total_amount = amount
                    print(record, datetime_transaction, price, total_amount, symbol)

                    db_stsm = (
                        db.insert(processed_price).values(price=price, amount=total_amount, time=raw_time, fk_currency_pair=id_pair, fk_exchange=1)
                    )
                    connection.execute(db_stsm)
                    record += 1

                    last_tid = tid
                    last_datetime_transaction = datetime_transaction
