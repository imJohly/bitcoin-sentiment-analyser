from operator import index
import pandas as pd

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import emoji
import re

from tqdm import tqdm

tqdm.pandas()

CSV_FILEPATH = './data/bitcoin_tweets.csv'

def import_data() -> pd.DataFrame:
    print(f"Importing CSV data from {CSV_FILEPATH} ...")
    try:
        df: pd.DataFrame = pd.read_csv(CSV_FILEPATH,
            encoding='utf-8',
            index_col=0,
            parse_dates=True,
        )

        # df: pd.DataFrame = pd.read_csv(CSV_FILEPATH, encoding='utf-8', index_col=0)
        print(df.columns.values)
        return df
    except FileNotFoundError as e:
        print(f"{e}: Couldn't find an existing CSV...")

def tokenise_and_lemmatise(text: str) -> str:
    tokens: list[str] = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]

    lemmatiser = WordNetLemmatizer()
    lemmatised_tokens = [lemmatiser.lemmatize(token) for token in filtered_tokens]

    processed_text = ' '.join(lemmatised_tokens)    
    return processed_text

analyser = SentimentIntensityAnalyzer()
def get_sentiment(text: str) -> int:
    scores = analyser.polarity_scores(text)
    sentiment = 1 if scores['pos'] > 0 else 0
    return sentiment

def data_pre_process(df: pd.DataFrame) -> pd.DataFrame:
    # Filter out any unverified tweets that may come from bots
    # df = df[df['user_verified'] == True]
    # print(f"Filtered out user_verified - DF length: {len(df)}")

    # Filter out any retweets to reduce the chance of data duplication bias
    df = df[df['is_retweet'] == False]
    print(f"Filtered out is_retweet - DF length: {len(df)}")

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

def main() -> None:
    print("Data pipeline starting...")

    df: pd.DataFrame = import_data()

    # Grab relevant columns
    df = df[['date', 'text', 'is_retweet', 'user_verified']]

    df = data_pre_process(df)

    # Sentiment analysis of dataframe
    df['sentiment'] = (
            df['text']
            .astype(str)
            .progress_apply(get_sentiment))

    # Group data by month and day
    print('Grouping data by day')
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.groupby(['month', 'day'], as_index=False)['sentiment'].sum()

    # Save processed data into out.csv
    df.to_csv('./out/out.csv') 
        
if __name__ == "__main__":
    main()
