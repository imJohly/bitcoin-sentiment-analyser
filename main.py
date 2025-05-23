import os
import pandas as pd
import numpy as np

from scipy.stats import zscore

import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from pmdarima import auto_arima

import emoji
import re

from tqdm import tqdm

tqdm.pandas()

CSV_FILEPATH = './data/bitcoin_tweets1000000.csv'

def import_data() -> pd.DataFrame:
    print(f"Importing CSV data from {CSV_FILEPATH} ...")
    try:
        df: pd.DataFrame = pd.read_csv(CSV_FILEPATH,
            encoding='utf-8',
            index_col=0,
            parse_dates=True,
        )
        return df
    except FileNotFoundError as e:
        print(f"{e}: Couldn't find an existing CSV...")
        quit()

def tokenise_and_lemmatise(text: str) -> str:
    tokens: list[str] = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    lemmatiser = WordNetLemmatizer()
    lemmatised_tokens = [lemmatiser.lemmatize(token) for token in filtered_tokens]

    processed_text = ' '.join(lemmatised_tokens)    
    return processed_text

def data_pre_process(df: pd.DataFrame) -> pd.DataFrame:
    # Filter out any unverified tweets that may come from bots
    # df = df[df['user_verified'] == True]
    # print(f"Filtered out user_verified - DF length: {len(df)}")

    # Filter for sources from 'Twitter for Android/iPhone/Mac/iPad'
    print("Filtering out unreliable sources...")
    allowed_sources = [
        'Twitter for Android',
        'Twitter Web App',
        'Twitter for iPhone',
        'Twitter for iPad',
        # 'Twitter for Mac',
    ]
    df = df[df['source'].isin(allowed_sources)]

    # Filter out any retweets to reduce the chance of data duplication bias
    print(f"Filtering out retweets...")
    df = df[df['is_retweet'] == False]

    print("Filtering out emojis...")
    df['text'] = df['text'].progress_apply(lambda s: emoji.replace_emoji(s, ''))

    print("Filtering out special characters...")
    df['text'] = df['text'].progress_apply(lambda s: re.sub(r'[^a-zA-Z0-9\s]', '', str(s)))

    print("Filtering out any hyperlinks....")
    df['text'] = df['text'].progress_apply(lambda s: re.sub('http[^\s]*', '', str(s)))

    print("Tokenising words in tweets")
    df['text'] = df['text'].progress_apply(tokenise_and_lemmatise)

    return df


analyser = SentimentIntensityAnalyzer()
def get_sentiment(text: str) -> int:
    scores = analyser.polarity_scores(text)
    # sentiment = 1 if abs(scores['compound']) > 0.5 else 0
    if scores['compound'] > 0.5:
        sentiment = 1
    elif scores['compound'] < -0.5:
        sentiment = -1
    else:
        sentiment = 0

    return sentiment

def aggregate_by_date(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.groupby(['year', 'month', 'day'], as_index=False)['sentiment'].sum()
   
    # Combine separated year, month, day columns into date again
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df[['date', 'sentiment']]

    # re-index and fill in missing dates with NaN
    df = df.set_index('date').asfreq('D').reset_index()

    df['sentiment'] = df['sentiment'].interpolate(method='linear')

    return df

def normalise_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # Z-score normalisation
    df['zscore'] = (df[column] - df[column].mean()) / df[column].std()

    # Z-score filtering
    THRESHOLD = 2
    df = df.mask(df['zscore'].abs() >= THRESHOLD).interpolate()

    return df

def cross_correlation(series_1: pd.Series, series_2: pd.Series) -> float:
    s1_np = series_1.to_numpy()
    s2_np = series_2.to_numpy()

    return np.corrcoef(s1_np, s2_np)[0, 1]

def get_diff_order_for_stationarity(series, signif=0.05, max_diff=5):
    d = 0
    current_series = series.copy()

    for i in range(max_diff + 1):
        adf_result = adfuller(current_series.dropna(), autolag='AIC')
        p_value = adf_result[1]
        print(f'Differencing level {d}: p-value = {p_value:.4f}')

        if p_value < signif:
            print(f'Stationary at differencing level {d}')
            return d
        else:
            current_series = current_series.diff()
            d += 1

    print('Series is still non-stationary after max differencing.')
    return d

# ---

def main() -> None:
    print("Data pipeline starting...")
    
    # Import bitcoin data
    b_df = pd.read_csv('./data/bitcoin_historical_data.csv',
                       encoding='utf-8',
                       index_col=0,
                       parse_dates=True,
                       dayfirst=True)

    if not os.path.isfile('./out/preprocess_out.csv'):
        df = import_data()

        # Grab relevant columns
        df = df[['date', 'text', 'source', 'is_retweet', 'user_verified']]

        df = data_pre_process(df)

        # save out preprocessed data to speed up re-runs
        df.to_csv('./out/preprocess_out.csv')
    else:
        df: pd.DataFrame = pd.read_csv('./out/preprocess_out.csv',
            encoding='utf-8',
            index_col=0,
        )
        print('Preprocessed data exists! Skipping...')

    # Sentiment analysis of dataframe
    if not os.path.isfile('./out/raw_sentiment.csv'):
        print('Scoring data by sentiment')
        df['sentiment'] = (
            df['text']
            .astype(str)
            .progress_apply(get_sentiment)
        )

        # Save raw sentiments
        df.to_csv('./out/raw_sentiment.csv')
    else:
        df: pd.DataFrame = pd.read_csv('./out/raw_sentiment.csv',
            encoding='utf-8',
            index_col=0,
        )
        print('Calculated sentiment data exists! Skipping...')

    print('Grouping data by day')
    df = aggregate_by_date(df)

    print('Normalising data and filtering out outliers via z-score')
    df = normalise_data(df, 'sentiment')
    b_df = normalise_data(b_df, 'Price')

    # calculate the cross-correlation
    print('Calculating cross correlation between bitcoin price and sentiment')
    correlation = cross_correlation(df['zscore'], b_df['zscore'])
    print(f'Cross correlation: {correlation}')

    # combine price and sentiment dataframes
    df['price'] = b_df.reset_index()['Price'][::-1].reset_index(drop=True)
    df['zscore2'] = b_df.reset_index()['zscore'][::-1].reset_index(drop=True)

    # Granger causality test
    _ = grangercausalitytests(df[['zscore', 'zscore2']], maxlag=4, verbose=True)

    # ARIMA
    d = get_diff_order_for_stationarity(df['zscore'])

    # Fit model
    model = auto_arima(df['zscore'], d=d,
                    seasonal=False,
                    trace=True,
                    error_action='ignore',
                    suppress_warnings=True)

    # Forecast next 10 steps
    n_periods = 10
    forecast = model.predict(n_periods=n_periods)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(df['zscore'], label='Original')
    plt.plot(range(len(df['zscore']), len(df['zscore']) + n_periods), forecast, label='Forecast', linestyle='--', color='orange')
    plt.title('auto_arima Forecast')
    plt.legend()
    plt.show()

    # # Save processed data into out.csv
    # df.to_csv('./out/out.csv') 

    # # plot
    # plt.plot(df['date'], df['zscore'], label='bitcoin sentiment')
    # plt.plot(df['date'], df['zscore2'], label='bitcoin price')
    # plt.xlabel('Date')
    # plt.ylabel('zscore')
    #
    # plt.show()


if __name__ == "__main__":
    main()
