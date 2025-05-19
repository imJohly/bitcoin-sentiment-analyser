import os
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
        'Twitter for iPhone',
        'Twitter for iPad',
        'Twitter for Mac',
    ]
    df = df[df['source'].isin(allowed_sources)]

    # Filter out any retweets to reduce the chance of data duplication bias
    print(f"Filtering out retweets...")
    df = df[df['is_retweet'] == False]

    # Remove any emoji characters
    print("Filtering out emojis...")
    df['text'] = df['text'].progress_apply(lambda s: emoji.replace_emoji(s, ''))

    # Remove any special characters
    print("Filtering out special characters...")
    df['text'] = df['text'].progress_apply(lambda s: re.sub(r'[^a-zA-Z0-9\s]', '', str(s)))

    # Remove any https
    print("Filtering out any hyperlinks....")
    df['text'] = df['text'].progress_apply(lambda s: re.sub('http[^\s]*', '', str(s)))

    # Tokenise text for sentiment analysis
    print("Tokenising words in tweets")
    df['text'] = df['text'].progress_apply(tokenise_and_lemmatise)

    return df

analyser = SentimentIntensityAnalyzer()
def get_sentiment(text: str) -> int:
    scores = analyser.polarity_scores(text)
    sentiment = 1 if abs(scores['compound']) > 0.5 else 0
    return sentiment

def aggregate_by_date(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.groupby(['year', 'month', 'day'], as_index=False)['sentiment'].sum()

    return df

def main() -> None:
    print("Data pipeline starting...")
    
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
            parse_dates=True,
        )
        print('Preprocessed data exists! Skipping...')

    # Sentiment analysis of dataframe
    print('Scoring data by sentiment')
    df['sentiment'] = (
        df['text']
        .astype(str)
        .progress_apply(get_sentiment)
    )

    # Group data by month and day
    print('Grouping data by day')
    df = aggregate_by_date(df)

    # Save processed data into out.csv
    df.to_csv('./out/out.csv') 
        
if __name__ == "__main__":
    main()
