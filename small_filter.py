import pandas as pd

CSV_FILEPATH = './data/bitcoin_tweets1000000.csv'

df: pd.DataFrame = pd.read_csv(CSV_FILEPATH,
    encoding='utf-8',
    index_col=0,
    parse_dates=True,
    # dtype={'user_favourites' : pd.Int32Dtype},
)

df = df[df['user_verified'] == True]

df.to_csv('./out/is_verified.csv')
